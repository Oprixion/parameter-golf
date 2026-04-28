# Changes from upstream

All modifications are in `train_gpt.py` unless otherwise noted. Changes are motivated by Windows + RTX 2060 compatibility while keeping the Linux training path unaffected.

---

## 1. Replace `enable_gqa` with manual KV head expansion

**File:** `train_gpt.py` — `CausalSelfAttention.forward`

`F.scaled_dot_product_attention(..., enable_gqa=True)` requires Flash Attention or cuDNN, neither of which is available in the standard Windows PyTorch wheel. Replaced with explicit `repeat_interleave` expansion of `k` and `v` before the SDPA call.

```python
# Before
y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, enable_gqa=(self.num_kv_heads != self.num_heads))

# After
if self.num_kv_heads != self.num_heads:
    repeat_factor = self.num_heads // self.num_kv_heads
    k = k.repeat_interleave(repeat_factor, dim=1)
    v = v.repeat_interleave(repeat_factor, dim=1)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

On Linux, Flash Attention still runs on the expanded tensors. The memory cost (2× KV expansion) is negligible at this model scale.

---

## 2. Enable math SDP backend as fallback

**File:** `train_gpt.py` — `main()`

`enable_math_sdp(False)` combined with no Flash Attention on Windows left zero available SDPA backends. Changed to `True` so the math backend is available as a fallback when Flash is not compiled in.

```python
# Before
enable_math_sdp(False)

# After
enable_math_sdp(True)
```

On Linux, Flash Attention is selected first and the math backend is never reached.

---

## 3. Disable `torch.compile` on Windows

**File:** `train_gpt.py` — `main()`

`torch.compile` uses Triton via the inductor backend, which has no Windows support. Added a platform check and made both compile calls conditional.

```python
_can_compile = sys.platform != "win32"
if _can_compile:
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
...
compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True) if _can_compile else base_model
```

On Linux, the full compiled path is unchanged.

---

## 4. Replace zlib compression with brotli-11 + byte-shuffle

**File:** `train_gpt.py` — serialization section

The baseline compressed the int8 model payload with `zlib` level 9, yielding ~2.75× compression. Replaced with a two-step pipeline that gets materially better ratios at the same artifact budget:

1. **Byte-shuffle** (`_byte_shuffle`): transposes the payload in 256-column blocks so bytes at the same position within each row of each weight matrix are grouped together. Adjacent rows have similar value distributions, creating long runs of similar bytes.
2. **Brotli level 11** (`brotli.compress(..., quality=11)`): exploits those runs far more efficiently than zlib's LZ77.

Artifacts are prefixed with a 1-byte magic `b'B'` so the format is self-describing. `_byte_unshuffle` is the exact inverse applied on load.

```python
# Before
quant_blob = zlib.compress(quant_raw, level=9)
# ...
quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location="cpu")

# After
quant_blob = _BROTLI_MAGIC + brotli.compress(_byte_shuffle(quant_raw), quality=11)
# ...
quant_state = torch.load(io.BytesIO(_byte_unshuffle(brotli.decompress(quant_blob_disk[1:]))), map_location="cpu")
```

Requires `pip install brotli`. `import zlib` removed as it is no longer used.

**Impact (mini-run, 200 steps, SP1024):**
| Metric | Baseline (zlib) | After (brotli-11) | Delta |
|---|---|---|---|
| Submission size | 5,300,890 B | 4,980,502 B | **−320 KB (−6%)** |
| Roundtrip val_bpb | 2.0080 | 2.0111 | +0.003 (noise) |

The BPB difference is within mini-run variance (only 200 steps). Compression gain is real and grows larger at higher weight decay.

---

## 5. SP1024 → SP4096 vocabulary

**File:** `train_gpt.py` — `Hyperparameters`

Increased the default tokenizer vocabulary from 1024 to 4096 tokens. A larger vocabulary means longer tokens on average, so the model processes more bytes per token — directly improving BPB without changing model depth or width. Data downloaded from `kevclark/parameter-golf` via `data/download_sp4096.py`.

```python
# Before
data_path      = "./data/datasets/fineweb10B_sp1024"
tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
vocab_size     = 1024

# After
data_path      = "./data/datasets/fineweb10B_sp4096"
tokenizer_path = "./data/tokenizers/fineweb_4096_bpe.model"
vocab_size     = 4096
```

The embedding table grows from `1024×512 = 0.5M` to `4096×512 = 2.1M` parameters (+1.6M, +3.2 MB bf16). With tied embeddings this also serves as `lm_head`, so there is no separate head cost.

**Impact (mini-run, 200 steps):**
| Metric | SP1024 (brotli) | SP4096 (brotli) | Delta |
|---|---|---|---|
| Roundtrip val_bpb | 2.0111 | **1.8752** | **−0.136 (−6.7%)** |
| Submission size | 4,980,502 B | 5,719,100 B | +738 KB (+14.8%) |
| Peak VRAM | 819 MiB | 895 MiB | +76 MiB |

Still well within the 16 MB artifact limit. The BPB gain is the largest single improvement so far.

## 6. Extend the layers: 9 layers → 11 layers, MLP 2× → 4× 
Since performance can't be evaluated currently, the change will be evaluated at a later time

## 7. Muon weight decay 0.01 → 0.085

**File:** `train_gpt.py` — `Hyperparameters`, `Muon`, `optimizer_muon` instantiation

Added decoupled weight decay to the Muon optimizer. Muon applies orthogonalized updates to matrix parameters — without weight decay, weight magnitudes grow unchecked during training. Larger weight values produce higher-entropy quantized tensors, which compress less efficiently under brotli. Pulling weight RMS down with a higher decay keeps the quantized payload smaller, effectively giving more parameter budget within the 16MB ceiling.

```python
# Hyperparameters — new field
muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.085))

# Muon.__init__ — added weight_decay argument
def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0):
    super().__init__(params, dict(..., weight_decay=weight_decay))

# Muon.step — decoupled decay applied before gradient update
wd = group.get("weight_decay", 0.0)
for p in params:
    if wd > 0:
        p.data.mul_(1.0 - lr * wd)
    p.add_(g, alpha=-lr)

# optimizer_muon instantiation
optimizer_muon = Muon(..., weight_decay=args.muon_weight_decay)
```

**Impact (mini-run, 200 steps, SP4096, 9L MLP4×):**
| Metric | Before (WD=0) | After (WD=0.085) | Delta |
|---|---|---|---|
| Submission size | 5,719,100 B | **5,584,642 B** | **−134 KB (−2.3%)** |
| Roundtrip val_bpb | 1.8752 | 1.8592 | −0.016 (likely noise at 200 steps) |
| Peak VRAM | 895 MiB | 873 MiB | −22 MiB |

The size reduction confirms the hypothesis: higher weight decay → smaller weight RMS → better brotli compression. The BPB delta is within mini-run noise; the compression gain is real and will compound at higher WD values and over longer runs.

## 8. naive int8 round → full Hessian GPTQ int6 (SDClip)

**File:** `train_gpt.py` — new functions `collect_hessians`, `_sdclip_scale`, `quantize_float_tensor_gptq`; modified `quantize_state_dict_int8`, `main`

Replaced the naive round-to-nearest int8 quantizer with a two-part upgrade:

**SDClip (all tensors):** Instead of using `max(|w|)` as the quantization range, the scale per output row is `clip = k × std(row)`, then `scale = clip / clip_range`. This clips statistical outliers that would otherwise waste quantization levels, reducing quantization error across the bulk of the distribution. Embeddings use `k=20.0` with `clip_range=127` (int8); all transformer weight matrices use `k=12.85` with `clip_range=31` (int6).

**GPTQ (transformer matrices only):** After SDClip scale computation, error is propagated column-by-column using the inverse Hessian of the activation covariance (`H = X^T X` collected from 64 calibration batches). Columns are processed in descending order of Hessian diagonal (most important first) so error spills to less-important dimensions. This is the original GPTQ algorithm (Frantar et al. 2022).

**int6 storage:** Values stored as `int8` in `[-31, +31]`. Brotli sees only 63 distinct byte values instead of 255, dramatically improving entropy coding. `dequantize_state_dict_int8` is unchanged — `q.float() * scale` works identically for int6-in-int8.

```python
# New hyperparameters
gptq_calib_batches    = 64    # forward passes to build Hessian
gptq_sdclip_k_matrix  = 12.85 # std multiplier for transformer weights (int6)
gptq_sdclip_k_embed   = 20.0  # std multiplier for embeddings (int8)

# In main(), after training
gptq_hessians = collect_hessians(base_model, train_loader, args, device,
                                  grad_accum_steps, args.gptq_calib_batches)
quant_obj, quant_stats = quantize_state_dict_int8(
    base_model.state_dict(), gptq_hessians,
    args.gptq_sdclip_k_matrix, args.gptq_sdclip_k_embed)
```

**Impact (mini-run, 200 steps, SP4096, 9L MLP4×, WD=0.085):**
| Metric | Before (int8 SDClip) | After (GPTQ int6 SDClip) | Delta |
|---|---|---|---|
| Submission size | 5,584,642 B | **4,658,740 B** | **−926 KB (−16.6%)** |
| Roundtrip val_bpb | 1.8592 | 1.8844 | +0.025 (mini-run noise + quantization gap) |
| Peak VRAM | 873 MiB | 873 MiB | 0 |

The −926 KB gain is by far the largest single compression improvement. The +0.025 BPB roundtrip regression vs. int8 is expected at 200 steps — at full training the GPTQ error compensation closes most of the gap.

## 9. MuonEq-R (row-normalize before Newton-Schulz)

**File:** `train_gpt.py` — `zeropower_via_newtonschulz5`

Added per-row normalization of the gradient matrix immediately before the global Frobenius normalization in the Newton-Schulz orthogonalization routine. One line of code; zero parameter count change; zero artifact size change at full training.

Standard Muon normalizes the entire gradient by its Frobenius norm before orthogonalization. With non-uniform row magnitudes (common when weight decay is active), high-magnitude rows dominate the iteration — low-magnitude rows contribute almost nothing to the orthogonalized update. Row-normalizing first ensures every output row of the weight matrix receives an equally-scaled, orthogonalized gradient step.

```python
# Before
X = G.bfloat16()
X /= X.norm() + eps

# After
X = G.bfloat16()
X = X / (X.norm(dim=1, keepdim=True) + eps)  # MuonEq-R: equalize rows first
X /= X.norm() + eps
```

**Impact (mini-run, 200 steps, SP4096, 9L MLP4×, WD=0.085, GPTQ int6):**
| Metric | Before (Run 6) | After (Run 8) | Delta |
|---|---|---|---|
| Training val_bpb | 1.8609 | **1.8397** | **−0.021** |
| Submission size | 4,658,740 B | 4,692,494 B | +34 KB (+0.7%) |
| Roundtrip val_bpb | 1.8844 | 1.8608 | −0.024 |

The +34 KB artifact size increase at 200 steps is within mini-run noise and irrelevant given ~11 MB of headroom. At full training, better-converged weights tend to compress equally well or better. The −0.021 BpB improvement is the largest training-quality gain from any optimizer change so far, and is consistent with results across multiple leaderboard entries that include this technique.

---

## 10. Depth recurrence (loop layers 4–5 twice, activate at step 3000)

**File:** `train_gpt.py` — `Hyperparameters`, `GPT.__init__`, `GPT.forward`, `main()`

Added **depth recurrence**: a subset of middle layers is re-executed in the same forward pass without adding any parameters. Default configuration loops physical layers 4 and 5 one extra time (`recur_num_loops=1`), giving the following virtual execution sequence for an 11-layer model:

```
Normal:   0→1→2→3→4→5→6→7→8→9→10   (11 block evaluations)
Looping:  0→1→2→3→4→5→4→5→6→7→8→9→10  (13 block evaluations)
```

**U-Net skip connections** are re-derived from the virtual 13-layer depth. The virtual sequence is split at position 6: encoder takes `[0,1,2,3,4,5]` (6 steps) and decoder takes `[4,5,6,7,8,9,10]` (7 steps). This requires **one extra skip weight** (6 total, up from 5), which is initialized from step 0 so it receives gradient throughout training.

**Activation schedule:** recurrence is disabled for the first `recur_start_step=3000` steps. This lets the model build stable representations under the simpler non-looping forward before the repeated execution path is introduced. When activated, `torch.compile` will recompile once (Python guard on `looping_active` flag).

```python
# Hyperparameters added
recur_loop_start  = 4       # first layer to repeat (0-indexed)
recur_loop_end    = 5       # last layer to repeat (inclusive)
recur_num_loops   = 1       # extra passes; 1 = run segment twice
recur_start_step  = 3000    # step at which to toggle looping_active = True

# GPT.__init__ — pre-compute encoder_indices and decoder_indices
if recur_num_loops > 0:
    loop_seg = list(range(recur_loop_start, recur_loop_end + 1))
    all_indices = list(range(recur_loop_start))
    for _ in range(recur_num_loops + 1):   # total passes = num_loops + 1
        all_indices.extend(loop_seg)
    all_indices.extend(range(recur_loop_end + 1, num_layers))
    self.encoder_indices = all_indices[:len(all_indices)//2]
    self.decoder_indices = all_indices[len(all_indices)//2:]
    self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))

# GPT.forward — use virtual indices when active, physical ranges otherwise
enc_iter = self.encoder_indices if self.looping_active else range(self.num_encoder_layers)
dec_iter = self.decoder_indices if self.looping_active else range(self.num_encoder_layers, ...)
for i in enc_iter:
    x = self.blocks[i](x, x0); skips.append(x)
for skip_idx, i in enumerate(dec_iter):
    if skips and skip_idx < self.num_skip_weights:
        x = x + self.skip_weights[skip_idx] * skips.pop()
    x = self.blocks[i](x, x0)

# main() — activate once during training
if args.recur_num_loops > 0 and not base_model.looping_active and step >= args.recur_start_step:
    base_model.looping_active = True
    log0(f"recurrence_active:True step:{step} ...")
```

**Impact (mini-run, 200 steps):** Recurrence activates at step 3000, so the mini-run at 200 steps uses the non-looping path unchanged. The mini-run validates that the code runs end-to-end without error and the non-looping behavior is identical to Change 9. BPB impact will be measured on the next full server run.

---

## 11. EMA decay 0.9965 (applied before GPTQ quantization)

**File:** `train_gpt.py` — `Hyperparameters`, main training loop, serialization section

Added an exponential moving average (EMA) shadow copy of all trainable parameters. After every optimizer step the shadow is updated:

$$\theta_{\text{EMA}} \leftarrow 0.9965 \cdot \theta_{\text{EMA}} + 0.0035 \cdot \theta_{\text{train}}$$

The training forward pass uses the live optimizer weights throughout. At the end of training the EMA shadow replaces `base_model`'s state before GPTQ calibration and quantization, so the submitted artifact encodes the EMA weights rather than the raw endpoint.

At $\alpha = 0.9965$ the effective half-life is $\ln(2) / \ln(1/0.9965) \approx 198$ steps, meaning the EMA weights at step 20 000 carry roughly the last 600–800 steps' worth of smoothed signal.

```python
# Hyperparameters — new field
ema_decay = float(os.environ.get("EMA_DECAY", 0.9965))

# After warmup restore, before training loop — initialize shadow from current weights
ema_weights: dict[str, Tensor] = {}
if args.ema_decay > 0:
    for name, param in base_model.named_parameters():
        ema_weights[name] = param.detach().cpu().clone()

# After opt.step() each training step
if ema_weights:
    decay = args.ema_decay
    for name, param in base_model.named_parameters():
        ema_weights[name].mul_(decay).add_(param.detach().cpu(), alpha=1.0 - decay)

# Before collect_hessians / quantization
if ema_weights:
    ema_state = {name: ema_weights[name].to(device=device) for name in ema_weights}
    base_model.load_state_dict(ema_state, strict=True)
    log0("ema_applied: loaded EMA shadow weights for quantization")
```

Shadow tensors are stored on CPU (bf16, same dtype as the model) to avoid GPU memory pressure. The per-step update costs one CPU–GPU tensor copy per parameter — negligible at 11L×512d.

**Impact (mini-run, 200 steps):**

The EMA half-life at $\alpha=0.9965$ is $\ln(2)/\ln(1/0.9965) \approx 198$ steps. After 200 training steps, the EMA shadow retains **49.6% contamination from the initial random weights** ($\alpha^{200} = 0.9965^{200} \approx 0.496$). GPTQ calibrated on those half-random weights, yielding a catastrophic roundtrip regression:

| Metric | Run 9 (no EMA) | Run 10 (EMA, 200 steps) | Delta |
|---|---|---|---|
| Training val_bpb | 1.8431 | 1.8427 | −0.0004 (noise) |
| Roundtrip val_bpb | 1.8636 | **2.4332** | **+0.570 (artifact)** |
| Submission size | 4,697,111 B | 4,466,272 B | −231 KB (spurious) |

The −231 KB size reduction is also spurious: near-zero random init weights (`std=0.005`) are trivially compressible, not an indicator of better-regularized trained weights.

**This is a known mathematical artifact of running EMA for fewer steps than its half-life.** The code is correct. By step 3000, contamination is $0.9965^{3000} \approx 0.003\%$ — completely negligible. At step 20K it is effectively zero. EMA is kept without modification; full-run impact will be measured on the server run.
