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

