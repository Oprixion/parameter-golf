# Run 1: Baseline
step:200/200 loss:3.4016 dt:187.8ms lr:4.00e-02 tok/s:348,964 train_time:38252ms step_avg:191.26ms eta:0s
step:200/200 val_loss:3.3896 val_bpb:2.0075 train_time:38295ms step_avg:191.48ms
peak memory allocated: 840 MiB reserved: 1286 MiB
Serialized model: 17114519 bytes
Code size: 48411 bytes
Total submission size: 17162930 bytes
Serialized model int8+zlib: 5252479 bytes (payload:6217360 raw_torch:6251745 payload_ratio:2.75x)
Total submission size int8+zlib: 5300890 bytes
final_int8_zlib_roundtrip val_loss:3.3905 val_bpb:2.0080 eval_time:5491ms
final_int8_zlib_roundtrip_exact val_loss:3.39047503 val_bpb:2.00803029

# Run 2: Replace zlib compression with brotli-11 + byte-shuffle
step:200/200 loss:3.4075 dt:187.8ms lr:4.00e-02 tok/s:348,892 train_time:38835ms step_avg:194.17ms eta:0s
step:200/200 val_loss:3.3935 val_bpb:2.0098 train_time:38863ms step_avg:194.32ms
peak memory allocated: 819 MiB reserved: 1298 MiB
Serialized model: 17114519 bytes
Code size: 49409 bytes
Total submission size: 17163928 bytes
Serialized model int8+brotli11: 4931093 bytes (payload:6217360 raw_torch:6251745 payload_ratio:2.75x)
Total submission size int8+brotli11: 4980502 bytes
final_int8_brotli11_roundtrip val_loss:3.3957 val_bpb:2.0111 eval_time:5487ms
final_int8_brotli11_roundtrip_exact val_loss:3.39569558 val_bpb:2.01112219

# Run 3: Update vocab size from 1024 to 4096
step:200/200 loss:4.2652 dt:185.4ms lr:4.00e-02 tok/s:353,538 train_time:42140ms step_avg:210.70ms eta:0s
step:200/200 val_loss:4.3128 val_bpb:1.8743 train_time:42212ms step_avg:211.06ms
peak memory allocated: 895 MiB reserved: 1632 MiB
Serialized model: 18687383 bytes
Code size: 49409 bytes
Total submission size: 18736792 bytes
Serialized model int8+brotli11: 5669691 bytes (payload:7009936 raw_torch:7044321 payload_ratio:2.66x)
Total submission size int8+brotli11: 5719100 bytes
final_int8_brotli11_roundtrip val_loss:4.3149 val_bpb:1.8752 eval_time:4706ms
final_int8_brotli11_roundtrip_exact val_loss:4.31489113 val_bpb:1.87520255

# Run 4: Extend the layers: 9 layers → 11 layers, MLP 2× → 4× 
N/A

# Run 5: Muon WD 0.01 → 0.085–0.090
step:200/200 loss:4.2289 dt:190.3ms lr:4.00e-02 tok/s:344,391 train_time:39642ms step_avg:198.21ms eta:0s
step:200/200 val_loss:4.2764 val_bpb:1.8585 train_time:39655ms step_avg:198.28ms
peak memory allocated: 873 MiB reserved: 1664 MiB
Serialized model: 18687383 bytes
Code size: 49705 bytes
Total submission size: 18737088 bytes
Serialized model int8+brotli11: 5534937 bytes (payload:7009936 raw_torch:7044321 payload_ratio:2.66x)
Total submission size int8+brotli11: 5584642 bytes
final_int8_brotli11_roundtrip val_loss:4.2781 val_bpb:1.8592 eval_time:4716ms
final_int8_brotli11_roundtrip_exact val_loss:4.27808402 val_bpb:1.8592066

# Run 6: naive int8 round → full Hessian GPTQ int6 (SDClip)
step:200/200 loss:4.2250 dt:176.1ms lr:4.00e-02 tok/s:372,092 train_time:37831ms step_avg:189.16ms eta:0s
step:200/200 val_loss:4.2820 val_bpb:1.8609 train_time:37847ms step_avg:189.23ms
peak memory allocated: 873 MiB reserved: 1664 MiB
Serialized model: 18687383 bytes
Code size: 54654 bytes
Total submission size: 18742037 bytes
Serialized model gptq-int6+brotli11: 4604086 bytes (payload:7009936 raw_torch:7044321 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4658740 bytes
final_gptq_int6_brotli11_roundtrip val_loss:4.3361 val_bpb:1.8844 eval_time:4717ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:4.33608348 val_bpb:1.88441250

# Run 7: Multi-head Latent Attention (MLA) (Reverted)
step:200/200 loss:4.3299 dt:188.6ms lr:4.00e-02 tok/s:347,578 train_time:39726ms step_avg:198.63ms eta:0s
step:200/200 val_loss:4.3848 val_bpb:1.9056 train_time:39754ms step_avg:198.77ms
peak memory allocated: 891 MiB reserved: 1644 MiB
Serialized model: 18690785 bytes
Code size: 55871 bytes
Total submission size: 18746656 bytes
Serialized model gptq-int6+brotli11: 4610311 bytes (payload:7009936 raw_torch:7047311 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4666182 bytes
final_gptq_int6_brotli11_roundtrip val_loss:4.4318 val_bpb:1.9260 eval_time:4760ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:4.43181055 val_bpb:1.92601440

# Run 8: MuonEq-R (row-normalize before Newton-Schulz)
step:200/200 loss:4.1823 dt:184.4ms lr:4.00e-02 tok/s:355,491 train_time:39772ms step_avg:198.86ms eta:0s
step:200/200 val_loss:4.2331 val_bpb:1.8397 train_time:39834ms step_avg:199.17ms
peak memory allocated: 873 MiB reserved: 1664 MiB
Serialized model: 18687383 bytes
Code size: 54802 bytes
Total submission size: 18742185 bytes
Serialized model gptq-int6+brotli11: 4637692 bytes (payload:7009936 raw_torch:7044321 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4692494 bytes
final_gptq_int6_brotli11_roundtrip val_loss:4.2818 val_bpb:1.8608 eval_time:4717ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:4.28182422 val_bpb:1.86083205

# Run 9: Depth recurrence: loop layers 4,5 ×2, activate at step ~2000 (RECUR_START_STEP=100 mini run)
step:200/200 loss:4.1798 dt:193.5ms lr:4.00e-02 tok/s:338,747 train_time:57616ms step_avg:288.08ms eta:0s
step:200/200 val_loss:4.2411 val_bpb:1.8431 train_time:57649ms step_avg:288.25ms
peak memory allocated: 1022 MiB reserved: 1682 MiB
Serialized model: 18688407 bytes
Code size: 57136 bytes
Total submission size: 18745543 bytes
Serialized model gptq-int6+brotli11: 4639975 bytes (payload:7010960 raw_torch:7045345 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4697111 bytes
final_gptq_int6_brotli11_roundtrip val_loss:4.2881 val_bpb:1.8636 eval_time:5597ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:4.28811519 val_bpb:1.86356603

# Run 10: EMA decay 0.9965
step:200/200 loss:4.1816 dt:198.3ms lr:4.00e-02 tok/s:330,474 train_time:45109ms step_avg:225.54ms eta:0s
step:200/200 val_loss:4.2402 val_bpb:1.8427 train_time:45114ms step_avg:225.57ms
peak memory allocated: 1022 MiB reserved: 1664 MiB
Serialized model: 18688407 bytes
Code size: 58124 bytes
Total submission size: 18746531 bytes
ema_applied: loaded EMA shadow weights for quantization
Serialized model gptq-int6+brotli11: 4408148 bytes (payload:7010960 raw_torch:7045345 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4466272 bytes
final_gptq_int6_brotli11_roundtrip val_loss:5.5990 val_bpb:2.4332 eval_time:5588ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:5.59895441 val_bpb:2.43324183

# Run 11: Set QK-Gain init = 4.0 for Conservative starting point
step:200/200 loss:4.1869 dt:195.6ms lr:4.00e-02 tok/s:335,021 train_time:61250ms step_avg:306.25ms eta:0s
step:200/200 val_loss:4.2293 val_bpb:1.8380 train_time:61255ms step_avg:306.28ms
peak memory allocated: 1022 MiB reserved: 1632 MiB
Serialized model: 18688407 bytes
Code size: 58124 bytes
Total submission size: 18746531 bytes
ema_applied: loaded EMA shadow weights for quantization
Serialized model gptq-int6+brotli11: 4410605 bytes (payload:7010960 raw_torch:7045345 payload_ratio:2.66x)
Total submission size gptq-int6+brotli11: 4468729 bytes
final_gptq_int6_brotli11_roundtrip val_loss:5.6568 val_bpb:2.4584 eval_time:5240ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:5.65682248 val_bpb:2.45839063

# Run 12: Full 20-min run — Changes 1–12 (4×H100, MAX_WALLCLOCK=1200s ≡ 8×H100 600s)
step:5832/20000 val_loss:2.7938 val_bpb:1.2142 train_time:1200122ms step_avg:205.78ms
stopping_early: wallclock_cap train_time:1200122ms step:5832/20000
peak memory allocated: 23304 MiB reserved: 23884 MiB
Serialized model: 131215522 bytes (131 MB float32)
Code size: 58125 bytes
ema_applied: loaded EMA shadow weights for quantization
Serialized model gptq-int6+brotli11: 13041574 bytes (payload_ratio:3.86x)
Total submission size gptq-int6+brotli11: 13099699 bytes (13.10 MB)
final_gptq_int6_brotli11_roundtrip val_bpb: 1.5208
Notes:
- Config: 11L, 512d, 8H/4KV, MLP×4, SP4096, tied embeddings, logit_softcap=30, RoPE
- Muon WD=0.085, MuonEq-R, EMA decay=0.9965, QK-gain init=4.0
- Depth recurrence: layers [4,5]×2, activated at step 3000 (within this run)
- GPTQ int6 SDClip (k_matrix=12.85, k_embed=20.0), 64 calib batches, brotli-11
- Quantization gap: +0.3066 BPB (1.2142 → 1.5208); likely due to recurrent layers
  being hard to calibrate with 64 batches after activation at step 3000
- Artifact 13.10 MB — within 16 MB limit

# Run 13: Full 20-min run + Vocab_Size = 8192
step:5028/20000 val_loss:2.9768 val_bpb:1.1524 train_time:1199808ms step_avg:238.63ms
stopping_early: wallclock_cap train_time:1199808ms step:5028/20000
peak memory allocated: 25372 MiB reserved: 27144 MiB
Serialized model: 135409826 bytes
Code size: 58147 bytes
Total submission size: 135467973 bytes
final_gptq_int6_brotli11_roundtrip val_loss:3.7854 val_bpb:1.4655 eval_time:3517ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:3.78541738 val_bpb:1.46545303

# Run 14: Full 20-min run + Vocab_Size = 8192, Parallel residuals — layers 7+ and 3 Layers recurrence
step:4979/20000 val_loss:2.9731 val_bpb:1.1510 train_time:1199987ms step_avg:241.01ms
stopping_early: wallclock_cap train_time:1199987ms step:4979/20000
peak memory allocated: 26310 MiB reserved: 27434 MiB
Serialized model: 135411874 bytes
Code size: 58998 bytes
Total submission size: 135470872 bytes
Total submission size gptq-int6+brotli11: 14157593 bytes
final_gptq_int6_brotli11_roundtrip val_loss:3.7864 val_bpb:1.4658 eval_time:4045ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:3.78642852 val_bpb:1.46584448

# Run 15: Progressive recurrence curriculum — loop [4,5]×3, frac-based activation
step:5476/20000 val_loss:2.9601 val_bpb:1.1460 train_time:2399992ms step_avg:438.27ms
stopping_early: wallclock_cap train_time:2399992ms step:5476/20000
peak memory allocated: 28162 MiB reserved: 29096 MiB
Serialized model: 135411874 bytes
Code size: 64756 bytes
Total submission size: 135476630 bytes
ema_applied: loaded EMA shadow weights for quantization
Serialized model gptq-int6+brotli11: 14146228 bytes (payload:36125024 raw_torch:36179596 payload_ratio:3.75x)
Total submission size gptq-int6+brotli11: 14210984 bytes
final_gptq_int6_brotli11_roundtrip val_loss:4.0333 val_bpb:1.5614 eval_time:8778ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:4.03326175 val_bpb:1.56140131
Notes:
- Hardware: 2×H100, MAX_WALLCLOCK_SECONDS=2400, QK_GAIN_INIT=5.25
- Config: SP8192, 11L, 512d, loop [4,5]×3 (phase1 at 50%, phase2 at 65% of wallclock)
  - recurrence_active:phase1 step:3217 frac:0.515
  - recurrence_active:phase2 step:3918 frac:0.665
- Parallel residuals: layers 7+, untie_loop_mlps=False
- GPTQ int6 SDClip (k_matrix=12.85, k_embed=20.0), 64 calib batches, brotli-11
- Quantization gap: +0.415 BPB (1.1460 → 1.5614) — worse than Run 14 (+0.315)
- Root cause: blocks [4,5] run 3× per forward pass; blended Hessian from 3 activation
  distributions makes GPTQ error compensation unreliable for shared weights
- SOTA (+0.012 gap) uses same architecture but 8×H100 — much better pre-quant model
  makes GPTQ error relatively smaller; cannot replicate at 2×H100 scale
- Experiment abandoned due to compute cost

# Run 16: GPTQ pipeline fix bundle (de-dup damping + k=5.0 + lm_head Hessian + untied loop MLPs)
step:5540/20000 val_loss:2.9568 val_bpb:1.1447 train_time:2399760ms step_avg:433.17ms
stopping_early: wallclock_cap train_time:2399760ms step:5540/20000
peak memory allocated: 24442 MiB reserved: 25496 MiB
Serialized model: 152188714 bytes
Code size: 66748 bytes
Total submission size: 152255462 bytes
ema_applied: loaded EMA shadow weights for quantization
Serialized model gptq-int6+brotli11: 15507569 bytes (payload:40327520 raw_torch:40384636 payload_ratio:3.77x)
Total submission size gptq-int6+brotli11: 15574317 bytes
final_gptq_int6_brotli11_roundtrip val_loss:3.7475 val_bpb:1.4508 eval_time:7074ms
final_gptq_int6_brotli11_roundtrip_exact val_loss:3.74753550 val_bpb:1.45078778
Notes:
- Hardware: 2×H100, MAX_WALLCLOCK_SECONDS=2400, QK_GAIN_INIT=5.25
- Config: SP8192, 11L, 512d, recur [4,5]×2 (RECUR_NUM_LOOPS=1), parallel_resid_start=7
- Untied loop MLPs: enabled (UNTIE_LOOP_MLPS=1) — 2 extra MLPs for the [4,5] repeat pass
- GPTQ fixes applied (Change 17): no double damping, k_matrix=5.0, lm_head Hessian collected
- Quantization gap: +0.306 BPB (1.1447 → 1.4508)
- Best post-quant result so far (vs Run 14 1.4658, Run 15 1.5614)
- Gap delta vs Run 14: −0.009 BPB (essentially noise — fixes did not close the structural gap)
- Pre-quant gain vs Run 15: −0.001 (noise); post-quant gain vs Run 15: −0.111 (untied MLPs +
  GPTQ fixes recovered most of the Run 15 progressive-curriculum regression)
- Artifact 15.57 MB — within 16 MB budget but tight; further capacity additions risk busting

## Verdict
- The 3 GPTQ pipeline bugs were real but not the dominant cause of the +0.30 gap. Fixes delivered
  ~−0.015 BPB, not the ~−0.18 BPB the previous diagnosis estimated.
- The gap appears structural at 2×H100 compute scale. Closing it requires research-level work
  (QAT, learned quantizers, group-wise scales, or different weight distributions by construction)
  rather than pipeline tuning.
- Best leaderboard-comparable result: 1.4508 BPB. Walking away from further iteration here.

# TODO

## Ranked by expected BPB impact (highest → lowest)

### 1. SP8192 vocabulary (~−0.025 BPB) ✅ Done
- Download dataset from `kevclark/parameter-golf` (fineweb10B_sp8192)
- Update `data_path`, `tokenizer_path`, `vocab_size = 8192` in Hyperparameters
- Embedding table: 8192×512 = 4.2M params → ~8.4MB bf16, still fits under 16MB with int6 GPTQ
- Blocker: dataset download required first

### 2. Parallel residuals — layers 7+ (~−0.010 BPB + quant gap reduction) ✅ Done
- GPT-J style: attention and MLP read from same pre-residual input at layers 7–10
- Zero parameter cost, fixes quantization gap by reducing activation interference during GPTQ calibration
- `Block.forward`: save `residual = x` before attn, compute `x = residual + attn(x) + mlp(residual)` for parallel layers

### 3. Extend recurrence to 3 layers: [3,4,5] (~−0.010 BPB) ✅ Done
- Currently looping [4,5]; SOTA loops [3,4,5] for 17 virtual layers from 11 physical
- Change `recur_loop_start = 3`, U-Net skip indices regenerate automatically

### 4. QK-gain 4.0 → 5.25 (~−0.005 BPB, trivial) ✅ Done
- Single env var: `QK_GAIN_INIT=5.25`
- Leaderboard shows monotonic improvement: 4.0 → 5.0 → 5.25; no code change needed

### 5. Partial RoPE: 16/64 head dims (~−0.003 BPB)
- Apply RoPE only to first 16 of 64 head dimensions; remaining 48 dims are position-agnostic
- Used in all SOTA records from 2026-03-21 onward

### 6. Layerwise LN scale (~−0.002 BPB)
- Per-layer learnable scalar on the RMSNorm output (one param per layer, AdamW group)
- Allows depth-dependent normalization strength; negligible parameter cost

### 7. Legal score-first TTT (~−0.002 BPB at eval time, most complex)
- Chunk val tokens into 32K-token chunks
- Phase 1: score all windows under `torch.no_grad()`; Phase 2: SGD on scored chunk
- SGD lr=0.005, momentum=0.9, 3 epochs per chunk, cosine LR decay, grad clip 1.0