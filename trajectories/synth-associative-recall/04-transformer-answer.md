**Problem.** BASED's spiky Taylor-2 kernel solved the short settings (0.988 / 0.972) but stayed at 0.180 at
`mqar-2048`. That long-context failure is not selectivity — the read is already maximally spiky — it is the
fixed-shape-state floor: recalling an *arbitrary* one of 128 bindings exactly needs carried state that grows
with the binding count, which a factored sub-quadratic kernel (feature dim 16) cannot provide. The fix is to
stop compressing the bindings into a fixed-shape state at all.

**Key idea (full causal attention).** Use the non-factoring similarity `exp(q^T k/√d)`. Because the
exponential tangles `q_i` and `k_j`, it forces the full `N×N` score matrix — but that is the point: every key
keeps its own column until the query arrives, so the bindings are never crushed. The exponential is also the
sharpest possible spike (no truncation, no approximation radius), so it is simultaneously the
maximally-selective read (for the short settings) and the no-compression read (for the long).

**Why it is the ceiling.** It pays `O(N²)` cost precisely to avoid the compression every sub-quadratic rung
accepted. The scaling `1/√d_k` (variance of `q^T k` is `d_k`) keeps the softmax off saturation so gradients
flow; `is_causal=True` masks future logits to `−∞`. Multi-head (2 heads, 32-dim each) gives independent
softmaxes so distinct relations stay distinct, at the cost of a single full-width head.

**Hyperparameters.** `num_heads=2`, `head_dim=32`; single bias-free `qkv` and `out_proj`; attention via
`F.scaled_dot_product_attention(..., is_causal=True)` (handles `1/√d_k` scaling and the causal mask). No
positional encoding or internal conv — the substrate's tied embeddings and fixed first-layer short causal
conv supply position; the mixer is a pure content-addressed lookup.

**Bar to clear.** Match BASED at 128/512 (little headroom) and, decisively, *hold* accuracy at `mqar-2048`
rather than collapsing to ~0.18 — a flat ~0.99 across length would confirm the sub-quadratic failures were
the state-capacity floor, and that keeping each binding addressable buys length-invariant recall.

```python
# EDITABLE region of custom_strategy.py — step 4: full causal multi-head attention (ceiling)
class CustomMixer(nn.Module):
    """Causal multi-head self-attention (full softmax attention)."""

    def __init__(self, d_model: int, seq_len: int, num_heads: int = 2):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)
```
