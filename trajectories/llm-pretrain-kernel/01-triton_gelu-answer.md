**Problem.** The FFN dominates the per-step compute of a GPT-2-class LM. Its two matmuls go to cuBLAS
near peak, but the activation, run as a chain of generic PyTorch ops, is a bandwidth-bound pass that
streams the wide `(tokens × 4·n_embd)` hidden through HBM over several elementwise launches and
temporaries. The floor to establish first is the activation everyone trusts — GELU — run *correctly*
and executed tightly, so any later win is attributable to a change I made, not a crippled baseline.

**Key idea.** Keep the scaffold's GELU (tanh approximation,
`0.5·x·(1 + tanh[√(2/π)·(x + 0.044715·x³)])`) but execute its whole elementwise chain in one custom
Triton kernel: each program instance owns a `BLOCK`-sized chunk of the flat hidden, masks the tail,
loads once, computes the polynomial and `tanh` in **fp32** (the cubic overflows / is coarse in bf16),
casts back, stores once. The two matmuls stay as ordinary torch `@` (cuBLAS) — this rung does *not*
hand-roll the matmul; it only collapses the activation's four-kernel chain into one launch.

**Why it is the floor.** Nothing about *what* the FFN computes changes — it is still GELU — so quality
lands at the unshaped-GELU level (asymptotically linear; on par with ReLU on LM perplexity). It is the
quality floor by construction, and competitive on throughput because it leaves the matmuls to cuBLAS.

**Backward.** Analytic, in a `torch.autograd.Function` (autograd can't trace the Triton kernel). The
two linears' grads are matmuls; the activation grad is the closed-form tanh-GELU derivative
`0.5·(1+tanh(inner)) + 0.5·x·sech²(inner)·c·(1+3·0.044715·x²)` evaluated in fp32 on the saved
pre-activation `h`. Limits check: derivative `→1` as `x→+∞`, `→0` as `x→−∞`, `=0.5` at `0`.

**Hyperparameters.** `BLOCK_SIZE = 1024` (power of two: enough work per instance to amortize launch and
saturate the bus, small enough for occupancy). `c = √(2/π) = 0.7978845608028654`, `a = 0.044715`. No
`CONFIG_OVERRIDES` — the loop's AdamW / cosine schedule is unchanged.

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — step 1: Triton fused GELU
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _fused_gelu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute entirely in float32 to avoid bfloat16 overflow in x^3
    xf = x.to(tl.float32)
    # tanh-approximation GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    inner = c * (xf + 0.044715 * xf * xf * xf)
    tanh_val = libdevice.tanh(inner)
    out = xf * 0.5 * (1.0 + tanh_val)
    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)

class _TritonGELUMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        h = x @ w_fc.t()
        act = torch.empty_like(h)
        n = h.numel()
        BLOCK = 1024
        grid = ((n + BLOCK - 1) // BLOCK,)
        _fused_gelu_kernel[grid](h, act, n, BLOCK_SIZE=BLOCK)
        out = act @ w_proj.t()
        ctx.save_for_backward(x, w_fc, w_proj, h, act)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, h, act = ctx.saved_tensors
        dtype = grad_output.dtype
        d_act = grad_output @ w_proj.to(dtype)
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ act.to(dtype).reshape(-1, act.shape[-1])
        # Analytical gradient of tanh-approximation GELU (matches the Triton forward)
        # gelu(x) = 0.5 * x * (1 + tanh(inner)), inner = c * (x + 0.044715 * x^3)
        # d_gelu/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * sech^2(inner) * d_inner/dx
        # d_inner/dx = c * (1 + 3 * 0.044715 * x^2)
        h_f = h.float()
        c = 0.7978845608028654
        inner = c * (h_f + 0.044715 * h_f * h_f * h_f)
        tanh_inner = torch.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        d_inner = c * (1.0 + 3.0 * 0.044715 * h_f * h_f)
        gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * h_f * sech2 * d_inner
        d_h = (d_act.float() * gelu_grad).to(dtype)
        grad_x = d_h @ w_fc.to(dtype)
        grad_w_fc = d_h.reshape(-1, d_h.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
        return grad_x, grad_w_fc, grad_w_proj

def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward with Triton fused GELU kernel."""
    return _TritonGELUMLP.apply(x, w_fc, w_proj)
```
