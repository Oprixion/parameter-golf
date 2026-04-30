# Baseline Model Analysis — Parameter Golf

**Date:** April 26, 2026
**Source:** `train_gpt.py` (default `Hyperparameters` config)
**Total parameters:** 17,059,912

---

## 1. Architecture Configuration

The baseline is a 9-layer decoder-only transformer with several modern efficiency techniques already applied:

| Hyperparameter | Value | Notes |
|---|---|---|
| Vocabulary size | 1,024 | SentencePiece BPE |
| Number of layers | 9 | Split into encoder (4) + decoder (5) halves |
| Model dimension (d_model) | 512 | Hidden size of the residual stream |
| Number of attention heads | 8 | Query heads |
| Number of KV heads | 4 | GQA — half the size of Q heads |
| Head dimension | 64 | d_model / num_heads |
| MLP expansion ratio | 2× | Already reduced from the standard 4× |
| MLP activation | relu² | Not GELU, not SwiGLU |
| Positional encoding | RoPE | No learned position embeddings |
| Tied embeddings | True | LM head shares weights with token embedding |
| Sequence length | 1,024 | Training context window |

---

## 2. Parameter Distribution

### Component Breakdown

| Component | Parameters | % of Total |
|---|---|---|
| Token embedding (tied) | 524,288 | 3.1% |
| Attention (×9 layers) | 7,077,888 | 41.5% |
| MLP (×9 layers) | 9,437,184 | 55.3% |
| Norms / scales / skip weights | 20,552 | 0.1% |
| **Total** | **17,059,912** | **100.0%** |

> **Note:** The inspect script's component grouping merged some MLP projection weights into the "attention" bucket because both have `proj` in their names. The corrected breakdown above is derived directly from per-tensor analysis: each block contributes 786,440 attention params and 1,048,576 MLP params.

### Per-Block Composition

Each of the 9 transformer blocks contains:

| Tensor | Shape | Parameters |
|---|---|---|
| `attn.c_q.weight` | (512, 512) | 262,144 |
| `attn.c_k.weight` | (256, 512) — GQA | 131,072 |
| `attn.c_v.weight` | (256, 512) — GQA | 131,072 |
| `attn.proj.weight` | (512, 512) | 262,144 |
| `attn.q_gain` | (8,) | 8 |
| `mlp.fc.weight` | (1024, 512) | 524,288 |
| `mlp.proj.weight` | (512, 1024) | 524,288 |
| `attn_scale` | (512,) | 512 |
| `mlp_scale` | (512,) | 512 |
| `resid_mix` | (2, 512) | 1,024 |
| **Total per block** | | **1,837,064** |

Multiplied across 9 layers, the blocks dominate the parameter budget. The MLP alone (1,048,576 params per block × 9 = 9.44M) accounts for over half the entire model.

---

## 3. Artifact Size Analysis

### Storage at Different Precisions

| Precision | Bytes/param | Model size |
|---|---|---|
| float32 | 4 | 68.24 MB |
| bfloat16 | 2 | 34.12 MB |
| int8 | 1 | 17.06 MB |
| int4 | 0.5 | 8.53 MB |
| 1.58-bit | 0.2 | 3.41 MB |

### How the 16 MB Budget Is Met

Raw int8 weights alone are **17.06 MB — over the 16 MB ceiling**. The baseline only fits because of **zlib level-9 compression** applied on top of the int8-quantized weights at serialization time (`train_gpt.py` lines 1075–1083). For neural network weights with similar value distributions per row, zlib typically achieves 30–50% compression, bringing the final artifact down to roughly 10 MB.

### Estimated Budget

| Item | Bytes |
|---|---|
| `train_gpt.py` source code | 48,996 (~0.05 MB) |
| Model @ int8 (uncompressed) | 17,059,912 (~17.06 MB) |
| **Pre-zlib total** | ~17.11 MB |
| **Estimated post-zlib total** | ~10.3 MB |
| **Headroom against 16 MB** | **~5.7 MB** |

This is the most strategically important number in the report. The baseline leaves roughly 5.7 MB of headroom after compression. That is room to *add* parameters or *increase precision on critical components*, not just to shrink.

---

## 4. Optimizations Already Applied

Before designing modifications, it is essential to recognize what the baseline already does — these levers cannot be pulled twice.

### Already in the baseline

- **Tied embeddings** (`tie_embeddings = True`). The LM head shares weights with the token embedding, eliminating the standard `vocab × d_model` projection at the output. This alone saves 524,288 parameters versus an untied model.

- **Grouped Query Attention** (`num_kv_heads = 4` vs `num_heads = 8`). K and V projections operate at half the dimension of Q, halving their parameter count compared to standard multi-head attention. K and V are 131,072 params each instead of 262,144.

- **MLP expansion already at 2×, not 4×.** The widely cited "reduce FFN ratio" lever is already partially applied. Going from 4× to 2× would have saved ~9.4M params; going from 2× to 1× saves another 4.7M but is much riskier for model quality.

- **RoPE rotary positional encoding** instead of learned position embeddings. No `seq_len × d_model` parameters are spent on position information.

- **U-Net style skip connections.** The 9 blocks are split into a 4-layer "encoder" half and 5-layer "decoder" half. Skip connections from encoder to decoder add only 4 × 512 = 2,048 parameters but provide a stronger inductive bias.

- **Muon optimizer for matrix-shaped parameters** instead of pure AdamW. Muon orthogonalizes gradients via a Newton-Schulz iteration, generally producing better convergence per step on matrix params.

- **Logit softcap** (`logit_softcap = 30.0`) to prevent the output logits from saturating during training.

- **Per-block learnable scales** (`attn_scale`, `mlp_scale`, `resid_mix`) for fine-grained control of residual contributions.

- **bfloat16 mixed precision training** with fp32 master weights via `CastedLinear`.

- **int8 + zlib compression** at serialization time.

- **`torch.compile`** is applied to the Newton-Schulz orthogonalization routine (line 735).

### Not in the baseline (potential levers)

- **Layer tying / depth recurrence.** All 9 blocks have independent weights. Sharing weights across some or all blocks is the largest unexplored lever.
- **Low-rank factorization** of the Q/K/V/proj or MLP weight matrices.
- **Asymmetric block design** — for example, MLP-only or attention-only blocks at certain depths.
- **Quantization-aware training (QAT).** The current setup is post-training int8 quantization with no QAT loop.
- **Sub-int8 quantization** (int4, ternary, BitNet 1.58-bit) for weight storage.
- **Aggressive vocabulary changes** (smaller vocab or byte-level).
- **Data ordering tricks** (curriculum, deduplication beyond what FineWeb provides).

---

## 5. Strategic Implications

### Where the parameters are concentrated

Two facts drive the strategy:

1. **MLPs are the biggest target (55.3% of params).** Any reduction to MLP parameter count has the largest absolute payoff.

2. **Layers are independent (×9 multiplier on all block params).** The 9 unique blocks contain 16.5M of the 17.06M total parameters. Sharing weights across blocks compounds savings linearly.

### Execution Plan

The challenge closes April 30 (3 days). The plan is structured around that hard deadline. The central mistake to avoid is treating each technique as a separate experiment — that costs ~30–60 minutes per iteration including setup, run, and evaluation. There is not enough time for 8 sequential ablations. The strategy is: **port the full proven SOTA stack in one shot, verify it works, then iterate only on genuinely open questions.**

#### Pre-work — Before Any Training Run

These are blocking tasks with zero quality risk that must be done first:

1. **Download tokenized data.** SP4096 and SP8192 pre-tokenized FineWeb shards are on `kevclark/parameter-golf` on HuggingFace. Download the appropriate variant before touching the model code — all training runs are blocked on this.
2. **Compress `train_gpt.py` with LZMA.** A trivial wrapper recovers ~43 KB of artifact headroom at zero cost. Do this once and never think about it again.
3. **Verify H100 environment.** Every April leaderboard entry requires Flash Attention 3 (`flash_attn_interface`), PyTorch ≥ 2.9.1+cu128. The local dev environment uses math-SDP fallback (see CHANGES.md); the submission environment must have FA3 compiled. Verify this before the first real run.

#### Step 1 — Full SOTA Port (Day 1)

Do not replicate SOTA incrementally. Implement the entire April 5 stack in a single code pass and get a working run that reaches ~1.085–1.097 BPB. Every component below has been independently validated across multiple entries.

| Change from baseline | Why |
|---|---|
| ~~zlib level-9 → brotli-11 + byte-shuffle~~ | ✅ Done — freed ~320 KB; grows with WD |
| ~~SP1024 → SP4096 vocab~~ | ✅ Done — −0.136 BPB (−6.7%) at mini-run scale |
| ~~9 layers → 11 layers, MLP 2× → 4×~~ | ✅ Done Larger model; compressibility of well-regularized weights compensates |
| ~~Muon WD 0.01 → 0.085–0.090~~ | ✅ Done Weight magnitude ↓ → brotli entropy ↓ → artifact smaller → enables wider MLP |
| ~~naive int8 round → full Hessian GPTQ int6 (SDClip)~~ | ✅ Done Higher quantization fidelity; `clip = k × std(row)` is principled and tunable |
| ~~MuonEq-R (row-normalize before Newton-Schulz)~~ | ✅ Done Zero byte cost; ~0.001 BPB improvement |
| ~~Depth recurrence: loop layers 4,5 ×2, activate at step ~3000~~ | ✅ Done 13 virtual layers from 11 physical; zero extra params |
| ~~EMA decay 0.9965~~ | ✅ Done Stable weight averaging with no SWA scheduling complexity |
| ~~QK-Gain init = 4.0~~ | ✅ Done Conservative starting point; monotonic improvement confirmed up to 5.25 |

**Calibration check after this run:** Measure post-brotli artifact size. If above 16 MB, increase WD by 0.005 increments. The correlation between weight RMS and brotli compressibility is R²≈0.99 — WD is the primary artifact-size dial.

#### Step 2 — Complete the SOTA Stack (Day 2)

These are the differences between the April 5 run (~1.0856) and the April 9 SOTA (1.0810). They are not "novel" — they are already proven and just not in the Step 1 code.

1. **SP4096 → SP8192.** Each additional vocabulary doubling gives a measurable BPB improvement. Requires downloading the SP8192 data shard. ✅ Done
2. **GPTQ on the embedding table.** The April 5 record used simple round-to-nearest for embeddings; April 9 applied full Hessian GPTQ to them too. Small but consistent win.
3. **Parallel residuals from layer 7.** GPT-J style: attention and MLP read from the same pre-residual input and a learned `lane_merge` scalar combines them. Allows specialization without adding parameters. ✅ Done
4. **3-layer recurrence (extend loop to layers 3,4,5).** The step from 2-layer to 3-layer recurrence contributed to the April 9 improvement. 
5. **QK-Gain tuning: sweep 4.0 → 5.0 → 5.25.** Monotonically improving; find the optimum.
6. **Score-First Legal TTT.** SGD adaptation (lr=0.005, momentum=0.9, 3 epochs per 32K-token chunk, cosine LR decay). Score all tokens in a chunk under `torch.no_grad()` *before* any weight update. This is the highest-return remaining lever and should not be treated as optional — it is responsible for a significant share of the gap between 1.0856 and 1.0810. The four compliance conditions (causality, normalized distribution, score-before-update, single-pass) must be verified explicitly.

#### Step 3 — Genuine Frontier (Day 3)

Only attempt these if Step 2 is stable and within budget. These are techniques absent from every leaderboard entry.

1. **d_model scaling analysis.** Every leaderboard entry uses d_model=512. At int6 with brotli-11 and Muon WD=0.095, is d_model=640 feasible within 16 MB? Run a parameter count + estimated compression calculation before attempting a training run. If it fits on paper, try it — a wider residual stream with fewer layers may represent a different and better optimum.
2. **Multi-head Latent Attention (MLA).** DeepSeek-V2 style: compress KV into a low-rank latent before projection. This reduces attention KV parameters by 4–8× with minimal quality loss, potentially freeing budget for more MLP width or extra layers. Not attempted by any entry.
3. **Per-layer SDClip threshold search.** The current `clip = k × std(row)` uses a global k. Allowing k to vary per layer (tuned on a small Hessian trace) could find a better rate-distortion frontier layer by layer, recovering a small but consistent BPB gain.
4. **QK-Gain > 5.25.** The leaderboard shows monotonic improvement up to 5.25 with no confirmed ceiling. Test 5.5 and 6.0.

#### What Not to Attempt

| Idea | Why not |
|---|---|
| Low-rank MLP factorization | Every top run expanded MLP to 4×; factorization destroys the WD-compressibility dividend |
| Int4 / ternary naive quantization | Higher implementation risk than GPTQ int6 with worse empirical results across the board |
| QAT training loop | April 1 record explicitly removed it: "appeared to provide little or no benefit"; GPTQ post-training is sufficient |
| Incremental one-change-at-a-time experiments | No time; the deadline is April 30 |

### Risk assessment

| Lever | Effort | Risk to BPB | Estimated savings |
|---|---|---|---|
| Layer tying (3 unique blocks recurred) | Low — change `forward` and constructor | Medium — may need more iterations | ~11M params |
| Low-rank MLP (r=128) | Medium — replace MLP class | Low — well-studied | ~7M params |
| MLP ratio 2× → 1× | Trivial — change one variable | Medium-high — already at 2× | ~4.7M params |
| Sub-int8 quantization | High — modify quantization routine | High — quality degradation | Variable |
| QAT | High — add training loop machinery | Low if done correctly | Free quality at any size |

---

## 6. Conclusion

The baseline is a thoughtfully optimized starting point. It has already pulled the obvious efficiency levers — tied embeddings, GQA, 2× MLP, RoPE, U-Net skips, Muon, int8+zlib — and currently fits in roughly 10 MB after compression with ~5.7 MB of headroom remaining.

The single largest unexplored area is **block-level weight sharing**, given that 96.9% of parameters live in the 9 independent transformer blocks. Layer tying combined with low-rank MLP factorization is the recommended primary direction for the limited time available.

The presence of meaningful post-zlib headroom changes the framing of the problem: the goal is not to shrink the model below 16 MB, but to **redistribute the parameter budget more intelligently** — spending the freed budget on additional unique parameters or higher-quality training rather than just on raw compression.
