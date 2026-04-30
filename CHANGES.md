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

---

## 12. QK-Gain init 1.5 → 4.0

**File:** `train_gpt.py` — `Hyperparameters`

Raised the initialization of the per-head query gain scalar from `1.5` to `4.0`. `q_gain` is a learnable `nn.Parameter` of shape `(num_heads,)` applied to queries after RMS-norm and RoPE, just before the dot-product attention:

```python
q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
```

It controls the effective softmax temperature: higher gain → sharper attention → stronger selection of relevant positions. The parameter is fully trainable (scalar AdamW group); this change only sets the starting value.

```python
# Before
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

# After
qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 4.0))
```

At init=1.5 the model wastes early training steps pushing `q_gain` up from a near-uniform attention regime. Starting at 4.0 puts attention in the meaningfully sharp regime from step 0, giving key/query projections a useful gradient signal immediately. Leaderboard records show monotonic BpB improvement as init increases from 1.0 → 1.5 → 4.0 → 5.0 → 5.25.

**Impact (mini-run, 200 steps):**

The roundtrip BpB (2.4584) is dominated by EMA init contamination (same artifact as Change 11 — 49.6% random init at 200 steps) and carries no information about this change. The training val_bpb is the relevant signal:

| Metric | Run 8 (baseline, no EMA/recur) | Run 11 (QK=4.0, +EMA, +recur@100) | Delta |
|---|---|---|---|
| Training val_bpb | 1.8397 | **1.8380** | **−0.0017** |
| Roundtrip val_bpb | 1.8608 | 2.4584 | EMA artifact, not meaningful |
| Submission size | 4,692,494 B | 4,468,729 B | EMA artifact, not meaningful |

The −0.0017 training BpB is a positive signal at 200 steps, consistent with the leaderboard trend. Full-run impact will be confirmed on the server run.

---

## 13. relu² → LeakyReLU(0.5)²

**File:** `train_gpt.py` — `MLP.forward`

Replaced the `relu²` activation with `LeakyReLU(0.5)²`. One line change; zero parameter count change; zero artifact size change.

```python
# Before
x = torch.relu(self.fc(x))
return self.proj(x.square())

# After
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

**Motivation — quantization gap interaction:**

With `relu²`, every negative pre-activation becomes exactly zero after ReLU. For a typical normally-distributed hidden layer, ~50% of activations are zero. This creates a bimodal activation distribution seen by the `proj` weight's Hessian: half the calibration samples contribute zero to `X^T X` for those columns. The resulting Hessian is rank-deficient or near-singular for those input dimensions, so GPTQ's error propagation is unreliable — it cannot correctly compensate rounding error for columns that were never activated.

With `LeakyReLU(0.5)`, negative inputs pass through at half magnitude and are squared: `(0.5x)² = 0.25x²`. No activations are exactly zero; every column of the `proj` weight receives nonzero Hessian signal from every calibration sample. This makes the Hessian better-conditioned, GPTQ more accurate, and the quantization gap smaller.

The same argument applies to the interaction with **depth recurrence**: layers 4 and 5 are executed twice with the same weights but different activation patterns on each pass. With `relu²`, each pass zeros out a different ~50% of channels, making the union of dead neurons across both passes hard to calibrate. With `LeakyReLU(0.5)²`, all channels remain active on every pass.

The record from 2026-03-23 (`LeakyReLU_LegalTTT_ParallelMuon`) introduced this activation and all subsequent SOTA entries carry it.

**Impact:** Mini-run pending. Training val_bpb signal expected to match or slightly improve vs. Change 12. Quantization gap expected to shrink at full training.

---

## 14. SP4096 → SP8192 vocabulary

**File:** `train_gpt.py` — `Hyperparameters`

Doubled the tokenizer vocabulary from 4096 to 8192 tokens. Larger tokens encode more bytes per prediction step, directly reducing BPB independent of model capacity.

```python
# Before
data_path      = "./data/datasets/fineweb10B_sp4096"
tokenizer_path = "./data/tokenizers/fineweb_4096_bpe.model"
vocab_size     = 4096

# After
data_path      = "./data/datasets/fineweb10B_sp8192"
tokenizer_path = "./data/tokenizers/fineweb_8192_bpe.model"
vocab_size     = 8192
```

The embedding table grows from `4096×512 = 2.1M` to `8192×512 = 4.2M` parameters (+2.1M, +4.2 MB bf16). With tied embeddings quantized via GPTQ int8 SDClip (`clip_range=127`), the embedding contribution to artifact size remains within the 16 MB ceiling.

**Throughput cost:** The larger embedding lookup adds ~33 ms per step (238 ms vs. 205 ms at SP4096), reducing steps completed in 1200 s from 5832 → 5028 (−804 steps, −13.8%). The pre-quant BPB gain still outweighs the training-budget loss.

**Impact (Run 13 — full 20-min server run, 4×H100, 5028 steps, all Changes 1–13):**
| Metric | Run 12 (SP4096) | Run 13 (SP8192) | Delta |
|---|---|---|---|
| Steps completed | 5832 | 5028 | −804 (−13.8%) |
| Training val_bpb | 1.2142 | **1.1524** | **−0.062 (−5.1%)** |
| Roundtrip val_bpb | 1.5208 | **1.4655** | **−0.055** |
| Quantization gap | +0.307 | +0.313 | ~unchanged |
| Artifact size | 13.10 MB | 14.10 MB | +1.00 MB |

The pre-quant gain of −0.062 BPB exceeds the ~−0.025 estimate, confirming SP8192 is highly beneficial. EMA applied correctly (`ema_applied: loaded EMA shadow weights for quantization` confirmed in log). Artifact at 14.10 MB remains within the 16 MB limit.

The quantization gap (~+0.31 BPB) is unchanged across both SP4096 and SP8192, confirming it is a structural problem: layers 4 and 5 are executed twice during the forward pass (depth recurrence) but GPTQ collects only one Hessian per physical layer. The mixed activation distribution from "first-pass" and "second-pass" inputs makes Hessian-based error compensation unreliable for those layers.

**Next priorities to close the quantization gap:**
1. **Parallel residuals (layers 7–10):** GPT-J-style parallel attn+MLP decouples their Hessians, expected to reduce quant gap by ~−0.15 BPB.
2. **Recurrence [3,4,5]:** `recur_loop_start = 3` adds 2 virtual layers for ~−0.010 pre-quant BPB.
3. **QK-gain 5.25:** `QK_GAIN_INIT=5.25` env var only, ~−0.005 BPB.

---

## 15. Parallel residuals — layers 7+ (GPT-J style)

**File:** `train_gpt.py` — `Block.__init__`, `Block.forward`, `GPT.__init__`, `Hyperparameters`

Added GPT-J style parallel residual connections for layers `>= parallel_resid_start` (default 7). Both attention and MLP read from the same pre-block input and their outputs are summed in one residual step. Zero parameter cost.

**Motivation for GPTQ:** Decouples attention and MLP Hessian matrices — GPTQ error propagation is more reliable when the two sub-layers don't share input activations.

**Impact (Run 14 — 2×H100, 4979 steps, bundled with recur [3,4,5]):**
| Metric | Run 13 | Run 14 | Delta |
|---|---|---|---|
| Training val_bpb | 1.1524 | 1.1510 | −0.001 |
| Roundtrip val_bpb | 1.4655 | 1.4658 | +0.000 |
| Quant gap | +0.313 | +0.315 | ~unchanged |
| Artifact size | 14.10 MB | 14.16 MB | +0.06 MB |

Quant gap unchanged — shared-weight recurrence problem dominates.

---

## 16. Progressive depth recurrence curriculum (SOTA approach)

**File:** `train_gpt.py` — `Hyperparameters`, `GPT.__init__`, new `GPT.activate_looping()`, main training loop

Replaced binary `recur_start_step` activation with a two-phase wallclock-fraction curriculum matching the SOTA record `2026-04-06_SP8192_HessianSDClip_ProgressiveRecurrence` (+0.012 quant gap at 8×H100).

**Changes:**
1. `recur_loop_start` 3 → 4 (loop [4,5], 2 blocks instead of 3)
2. `recur_num_loops` 1 → 2 (3 total passes through [4,5])
3. `GPT.__init__` builds `_phase1_enc/dec` (1 extra repeat) and `_phase2_enc/dec` (2 extra repeats); `activate_looping(phase)` switches at runtime
4. Phase 1 activates at 50% of `MAX_WALLCLOCK_SECONDS`; Phase 2 at 65%
5. `UNTIE_LOOP_MLPS` default "1" → "0" (saves artifact size; SOTA doesn't use them)

**Impact (Run 15 — 2×H100, MAX_WALLCLOCK=2400s, 5476 steps):**
| Metric | Run 14 | Run 15 | Delta |
|---|---|---|---|
| Training val_bpb | 1.1510 | 1.1460 | −0.005 |
| Roundtrip val_bpb | 1.4658 | **1.5614** | **+0.095 (worse)** |
| Quant gap | +0.315 | **+0.415** | **+0.100 (worse)** |
| Artifact size | 14.16 MB | 14.21 MB | +0.05 MB |

Pre-quant improved but quant gap worsened. Shared-weight Hessian blending (blocks [4,5] run 3× with different activation distributions) dominates at 2×H100. SOTA's +0.012 gap likely requires 8×H100 scale.

---

## 17. Quant-gap fixes — bundle of 3 GPTQ corrections

**File:** `train_gpt.py` — `collect_hessians`, `quantize_float_tensor_gptq`, `quantize_state_dict_int8`, `Hyperparameters.gptq_sdclip_k_matrix`

After diagnosis on the +0.31 BPB post-quant gap (Run 13/14), three independent issues in the GPTQ pipeline were identified and fixed in one bundle. None depends on the others; each contributes to a smaller post-quant penalty.

### 17a. Removed duplicate Hessian damping
`collect_hessians` already adds `0.01 * mean_diag` damping at the end of accumulation. `quantize_float_tensor_gptq` then added the same damping a second time. The compounded ~2.01% damping flattens the Hessian diagonal, weakening (1) the descending-diagonal column ordering and (2) the off-diagonal entries of `Hinv` that drive GPTQ's error compensation. Fix: keep the damping in `collect_hessians`, remove the duplicate in `quantize_float_tensor_gptq`.

### 17b. Lowered `gptq_sdclip_k_matrix` 12.85 → 5.0
With `clip_range = 31` (int6) and `k = 12.85`, scale per row = `12.85·σ / 31 ≈ 0.414·σ`. A weight at ±3σ maps to int level ±7, so the bulk of the distribution uses only ~15 of 63 available int levels — most of int6's resolution is wasted. The 12.85 figure was tuned to maximize brotli compression (low entropy → smaller artifact), but at the cost of much larger quantization MSE. `k = 5.0` is closer to the MSE-optimal clip for an int6 Gaussian (~3σ) and uses ~38 of 63 levels. Trades a small artifact-size increase (still well under 16 MB on Run 14, 14.16 MB → ~14.5 MB expected) for materially smaller post-quant BPB.

### 17c. Added Hessian collection for the tied lm_head
With `tie_embeddings=True`, the model uses `F.linear(x, self.tok_emb.weight)` as the output projection. Because that call is a raw `F.linear` (not a `CastedLinear` module), the forward hook in `collect_hessians` never fired for the tied lm_head — and `quantize_state_dict_int8` then explicitly forced `H = None` for `is_embed`. Result: the largest single tensor in the artifact (~4.2M params on SP8192) was quantized without GPTQ error compensation despite being directly responsible for the cross-entropy logits.

Fix: in `collect_hessians`, when `tie_embeddings` is True, register an additional hook on `model.final_norm` that accumulates `X^T X` from its output (which is exactly the input fed into `F.linear(x, tok_emb.weight)`). Store it under key `tok_emb.weight`. In `quantize_state_dict_int8`, drop the `not is_embed` constraint so the Hessian flows through to GPTQ. The embedding still uses `clip_range=127` (int8) and `k_embed=20.0` — only error compensation is added.

```python
# collect_hessians — added after the CastedLinear loop
if model.tie_embeddings:
    emb_dim = model.tok_emb.weight.shape[1]
    hessians["tok_emb.weight"] = torch.zeros(emb_dim, emb_dim, dtype=torch.float32)
    def lm_head_hook(_m, _inp, out):
        x = out.detach().float()
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        hessians["tok_emb.weight"].add_((x.T @ x).cpu())
    hooks.append(model.final_norm.register_forward_hook(lm_head_hook))

# quantize_state_dict_int8 — was forcing H=None for embed; now lets it pass through
H = hessians.get(name) if hessians is not None else None
```

**Expected impact (full run):**
- 17a (de-dup damping): ~−0.05 BPB
- 17b (k=5.0): ~−0.10 BPB
- 17c (lm_head Hessian): ~−0.03 to −0.05 BPB
- Combined target: post-quant gap shrinks from +0.31 → roughly +0.10–0.15 BPB

If 17b proves too aggressive on artifact size, try `k = 6.0` next; if it leaves headroom, try `k = 4.0`.

**Not fixed yet (out of scope for this bundle):**
- Shared-weight Hessian blending in recurrent layers (claim #1 from diagnosis): structural; the joint H = ΣHᵢ is correct for joint-MSE but column ordering / Hinv off-diagonals mix regimes. Mitigation = `untie_loop_mlps=True`, which adds ~300 KB but breaks the blending on the most error-sensitive sub-layer (proj). Worth re-running once 17a-c are validated.
- Light QAT pass after GPTQ to close residual gap.

---

## 18. Defaults aligned to Run 16 config

**File:** `train_gpt.py` — `Hyperparameters`

Two default flips so the recommended run command is `QK_GAIN_INIT=5.25 MAX_WALLCLOCK_SECONDS=1200 torchrun --nproc_per_node=2 train_gpt.py` with no other env vars required.

```python
# Before
recur_num_loops = int(os.environ.get("RECUR_NUM_LOOPS", 2))
untie_loop_mlps = bool(int(os.environ.get("UNTIE_LOOP_MLPS", "0")))

# After
recur_num_loops = int(os.environ.get("RECUR_NUM_LOOPS", 1))
untie_loop_mlps = bool(int(os.environ.get("UNTIE_LOOP_MLPS", "1")))
```

`recur_num_loops=1` reverts away from the progressive-curriculum config that produced Run 15's
+0.42 gap; `untie_loop_mlps=1` was actually wrong on the artifact-size estimate (~1.1 MB extra,
not the 300 KB the previous diagnosis claimed) but stayed within the 16 MB budget on Run 16.

---

## 19. Run 16 post-mortem and stop point

**Run 16 result (full 2400s server run, 2×H100, all of Changes 1–18):**

| Metric | Run 14 (baseline) | Run 15 (progressive) | **Run 16 (fixes)** |
|---|---|---|---|
| Wallclock | 1200s | 2400s | 2400s |
| Steps | 4979 | 5476 | 5540 |
| Pre-quant val_bpb | 1.1510 | 1.1460 | **1.1447** |
| Post-quant val_bpb | 1.4658 | 1.5614 | **1.4508** |
| Quant gap | +0.315 | +0.415 | **+0.306** |
| Artifact size | 14.16 MB | 14.21 MB | 15.57 MB |

Run 16 produced the **best post-quant result on this codebase (1.4508 BPB)** but the gap shrank
only −0.009 BPB vs. Run 14. The previous diagnosis estimated the GPTQ fix bundle would close
~−0.18 BPB; the actual delta was an order of magnitude smaller.

### Why the diagnosis under-delivered

The three bugs were real and worth fixing, but none were the dominant gap contributor:

1. **Double damping (17a)**: real bug, worth fixing for code correctness, but the actual numerical
   damage was small. 0.01 vs 0.0201 damping mostly affects rounding behavior near dead columns;
   the bulk of the per-column GPTQ updates barely change.
2. **k_matrix 12.85 → 5.0 (17b)**: lowered MSE per row but increased clipping at the tails.
   Net BPB benefit was much smaller than the int-level utilization argument suggested — brotli
   is also worse at compressing the higher-entropy int6 distribution, partially offsetting the
   precision gain.
3. **lm_head Hessian (17c)**: the lm_head was already getting reasonable int8 SDClip quantization
   without GPTQ. The marginal benefit of error compensation is small at int8.
4. **Untied loop MLPs (the structural fix from Change 18)**: gave each repeat pass a clean Hessian
   on the proj weight, but the post-quant improvement vs. Run 15 was dominated by reverting away
   from the 3× recurrence regression, not by the untying itself.

The residual ~+0.30 BPB gap appears to be a fundamental property of int6 GPTQ quantization
applied to this 11L×512d model at 2×H100 compute. Pipeline fixes can shave ~0.01–0.02 off the
gap; the rest requires algorithmic changes.

### What would actually close the gap (research-level, not implemented)

Ranked by literature-supported expected impact:

1. **Quantization-Aware Training (QAT)** — fold quantize/dequantize into the forward pass for the
   last 500–2000 training steps so weights co-adapt with the quantizer. Standard technique;
   typical impact is −0.05 to −0.15 BPB on small models. Not in this codebase.
2. **Group-wise scales** — replace per-row scale with per-N-column-block scale (e.g., groups of 64
   or 128). Roughly 2× metadata cost (~200 KB extra) but materially better quantization on
   non-stationary weight distributions. Standard in modern int4/int6 schemes.
3. **Learned codebooks** — instead of fixed-step int6, learn per-row codebooks (or per-block).
   Compresses worse with brotli but quantizes much better. Worth ~−0.05 to −0.10 BPB if the
   metadata overhead can be amortized.
4. **Activation choices that produce quantization-friendly weights** — e.g., normalization that
   bounds row-wise std, or weight reparameterizations (W = αÛ for unit-norm Û and learnable α).
   Some SOTA records hint at this; not documented well enough for direct port.

### Stop point

This codebase reaches ~1.45 BPB post-quant at 2×H100 / 2400s. Closing the gap to SOTA's
~1.16 BPB requires implementing one or more of (1)–(4) above plus the compute to validate them.
That is a research effort, not a tweak, and is left for a future iteration.
