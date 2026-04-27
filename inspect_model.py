"""
inspect_model.py
Run from the project root: python inspect_model.py
No GPU, no training, no data needed.
"""
import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ── Minimal stubs so we can import the model without the full env ──────────
# Paste the classes we need directly to avoid import-side-effects from main()

# ---------- copy from train_gpt.py ----------

CONTROL_TENSOR_NAME_PATTERNS = ("attn_scale","attn_scales","mlp_scale",
    "mlp_scales","resid_mix","resid_mixes","q_gain","skip_weight","skip_weights")

class CastedLinear(nn.Linear):
    def forward(self, x):
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    def forward(self, seq_len, device, dtype):
        return torch.zeros(1), torch.zeros(1)  # stub

def apply_rotary_emb(x, cos, sin): return x  # stub

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim,    bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim,    bias=False)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)

class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn      = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp       = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

class GPT(nn.Module):
    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,
                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,
                 rope_base, qk_gain_init):
        super().__init__()
        self.tie_embeddings = tie_embeddings
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        num_encoder = num_layers // 2
        num_decoder = num_layers - num_encoder
        num_skip    = min(num_encoder, num_decoder)
        self.skip_weights = nn.Parameter(torch.ones(num_skip, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList([
            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)

# ── Baseline hyperparameters (from Hyperparameters class) ─────────────────
VOCAB_SIZE    = 1024
NUM_LAYERS    = 9
MODEL_DIM     = 512
NUM_HEADS     = 8
NUM_KV_HEADS  = 4
MLP_MULT      = 2
TIE_EMBEDDINGS = True

model = GPT(
    vocab_size         = VOCAB_SIZE,
    num_layers         = NUM_LAYERS,
    model_dim          = MODEL_DIM,
    num_heads          = NUM_HEADS,
    num_kv_heads       = NUM_KV_HEADS,
    mlp_mult           = MLP_MULT,
    tie_embeddings     = TIE_EMBEDDINGS,
    tied_embed_init_std= 0.005,
    logit_softcap      = 30.0,
    rope_base          = 10000.0,
    qk_gain_init       = 1.5,
)

# ── Per-tensor breakdown ──────────────────────────────────────────────────
print("\n── Per-tensor detail ────────────────────────────────────────────────")
print(f"  {'Name':<55} {'Params':>9}  Shape")
print(f"  {'-'*55} {'-'*9}  -----")
for name, p in model.named_parameters():
    print(f"  {name:<55} {p.numel():>9,}  {tuple(p.shape)}")

# ── Component-level summary ───────────────────────────────────────────────
total = sum(p.numel() for p in model.parameters())

buckets = {"embedding": 0, "attention": 0, "mlp": 0, "norms_scales": 0, "other": 0}
for name, p in model.named_parameters():
    n = p.numel()
    if "tok_emb" in name or "lm_head" in name:
        buckets["embedding"] += n
    elif "attn" in name or "c_q" in name or "c_k" in name or "c_v" in name or "proj" in name or "q_gain" in name or "rotary" in name:
        buckets["attention"] += n
    elif "mlp" in name or "fc" in name:
        buckets["mlp"] += n
    elif any(x in name for x in ("norm","scale","resid_mix","skip_weight")):
        buckets["norms_scales"] += n
    else:
        buckets["other"] += n

print("\n── Component summary ────────────────────────────────────────────────")
print(f"  {'Component':<20} {'Params':>10}  {'%':>6}  {'@ int8':>9}  {'@ fp32':>9}")
print(f"  {'-'*20} {'-'*10}  {'-'*6}  {'-'*9}  {'-'*9}")
for k, v in buckets.items():
    pct = 100 * v / total if total else 0
    print(f"  {k:<20} {v:>10,}  {pct:>5.1f}%  {v/1e6:>8.3f}MB  {v*4/1e6:>8.3f}MB")
print(f"  {'-'*20} {'-'*10}  {'-'*6}  {'-'*9}  {'-'*9}")
print(f"  {'TOTAL':<20} {total:>10,}  {'100.0':>6}  {total/1e6:>8.3f}MB  {total*4/1e6:>8.3f}MB")

# ── Precision size table ──────────────────────────────────────────────────
print("\n── Model size at different precisions ───────────────────────────────")
for label, bpp in [("float32 (4B)", 4), ("bfloat16 (2B)", 2),
                   ("int8 (1B)", 1), ("int4 (0.5B)", 0.5), ("1.58-bit (0.2B)", 0.2)]:
    mb = total * bpp / 1e6
    bar = "█" * int(mb / 0.5)
    print(f"  {label:<18} {mb:>7.3f} MB  {bar}")

# ── Artifact budget ───────────────────────────────────────────────────────
LIMIT = 16_000_000
script_bytes = os.path.getsize("train_gpt.py") if os.path.exists("train_gpt.py") else 0
int8_uncompressed = total * 1
print("\n── Artifact budget (16,000,000 byte limit) ──────────────────────────")
print(f"  train_gpt.py script size:     {script_bytes:>10,} bytes  ({script_bytes/1e6:.3f} MB)")
print(f"  Model weights @ int8:         {int8_uncompressed:>10,} bytes  ({int8_uncompressed/1e6:.3f} MB)")
print(f"  Est. total (before zlib):     {script_bytes+int8_uncompressed:>10,} bytes  ({(script_bytes+int8_uncompressed)/1e6:.3f} MB)")
remaining = LIMIT - script_bytes - int8_uncompressed
print(f"  Headroom (zlib ~saves ~40%):  {remaining:>10,} bytes  ({remaining/1e6:.3f} MB)")
print(f"  {'WITHIN LIMIT ✓' if remaining > 0 else 'OVER LIMIT ✗'}")

# ── Per-layer attention breakdown ─────────────────────────────────────────
print("\n── Per-layer attention parameter count ──────────────────────────────")
head_dim = MODEL_DIM // NUM_HEADS
kv_dim   = NUM_KV_HEADS * head_dim
q_params  = MODEL_DIM * MODEL_DIM
k_params  = MODEL_DIM * kv_dim
v_params  = MODEL_DIM * kv_dim
o_params  = MODEL_DIM * MODEL_DIM
attn_total = q_params + k_params + v_params + o_params
print(f"  Q projection  ({MODEL_DIM} × {MODEL_DIM}):    {q_params:>8,}")
print(f"  K projection  ({MODEL_DIM} × {kv_dim})  GQA: {k_params:>8,}")
print(f"  V projection  ({MODEL_DIM} × {kv_dim})  GQA: {v_params:>8,}")
print(f"  O projection  ({MODEL_DIM} × {MODEL_DIM}):    {o_params:>8,}")
print(f"  Attn total per layer:         {attn_total:>8,}")
print(f"  Attn total × {NUM_LAYERS} layers:        {attn_total*NUM_LAYERS:>8,}")

print("\n── Per-layer MLP parameter count ────────────────────────────────────")
hidden   = MLP_MULT * MODEL_DIM
fc_p     = MODEL_DIM * hidden
proj_p   = hidden * MODEL_DIM
mlp_total = fc_p + proj_p
print(f"  fc   ({MODEL_DIM} × {hidden}):           {fc_p:>8,}   [expansion {MLP_MULT}×]")
print(f"  proj ({hidden} × {MODEL_DIM}):           {proj_p:>8,}")
print(f"  MLP total per layer:          {mlp_total:>8,}")
print(f"  MLP total × {NUM_LAYERS} layers:         {mlp_total*NUM_LAYERS:>8,}")

print("\n── Key architecture facts ───────────────────────────────────────────")
print(f"  Embedding tied (lm_head shared): {TIE_EMBEDDINGS}")
print(f"  GQA: {NUM_HEADS} query heads / {NUM_KV_HEADS} KV heads  (ratio {NUM_HEADS//NUM_KV_HEADS}×)")
print(f"  MLP activation: relu² (not GELU, not SwiGLU)")
print(f"  MLP expansion ratio: {MLP_MULT}× (NOT the usual 4×)")
print(f"  Positional encoding: RoPE (no learned position embeddings)")
print(f"  Has U-Net style skip connections: YES (encoder/decoder halves)")
print()