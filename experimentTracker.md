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