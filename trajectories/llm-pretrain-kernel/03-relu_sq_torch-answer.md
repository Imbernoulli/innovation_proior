**Problem.** The fused rung split clean: squared ReLU lowered quality (`val_loss` 2.2749 < GELU's
2.2868, `lambada_ppl` 66.88 < 68.2) but the hand-rolled Triton up-matmul tanked throughput (`elapsed`
30344 vs 20035, ~50% slower) — a naive tiled GEMM can't beat cuBLAS/`torch.compile` on these FFN
shapes, and the activation round-trip it deleted is a rounding error next to that gap. Keep the
activation that won; throw away the kernel that lost.

**Key idea.** Squared ReLU `act(z) = max(z, 0)²` in **pure torch**: `h = x @ w_fc.t()`,
`a = relu(h)²`, `out = a @ w_proj.t()`. Both matmuls go back to cuBLAS (the path the GELU floor used);
only the activation and its backward stay custom. Squared ReLU is **ReGLU with tied weights** (`max(0,
xW)·(xW) = relu(xW)²`) — the GLU multiplicative interaction captured with the **two** matrices the edit
surface gives, no third matrix `V`, no width shrink. Power 2 is the minimal super-linear rectified
polynomial (gradient `2·relu(z)` grows only linearly; higher powers overflow bf16). Rectify, not plain
`x²` (`x²` is even/non-monotonic, discards the sign). The gradient `2·relu(z)` is magnitude-aware —
strongly-firing units get larger gradients, where GELU/Swish saturate near 1 — which is the inductive
bias behind the perplexity gain.

**Why it should win.** The activation is identical to the fused rung, so quality should match/slightly
beat 2.2749 (only fp32-accumulator vs cuBLAS-bf16 floating-point differences). With both matmuls on
cuBLAS, throughput should collapse from 30344 toward the GELU floor's 20035 — a Pareto improvement over
the fused rung on both axes.

**Backward.** Hand-written `torch.autograd.Function`; save `(x, w_fc, w_proj, h, relu_h)`. `relu_h`
gives the `2·relu(h)` activation-grad factor and (squared) the activation value for `∂L/∂w_proj`. Cast
saved tensors/weights to `grad_output`'s dtype in the matmuls (no bf16/fp32 mix). The two linears' grads
are matmuls; `∂L/∂h = 2·relu(h) ⊙ (g @ w_proj)`.

**Hyperparameters.** None to tune — parameter-matched activation, the loop's existing AdamW (`β=(0.9,
0.95)`, `wd=0.1`, grad-clip 1.0) and cosine schedule unchanged. No `CONFIG_OVERRIDES`. (Saving `relu_h`
directly rather than recomputing it — as the fused rung did — is the only bookkeeping change.)

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — step 3: ReLU^2 (pure torch, custom autograd)
def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward with ReLU^2 activation via custom autograd."""

    class ReLUSquaredMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_fc, w_proj):
            h = x @ w_fc.t()
            relu_h = F.relu(h)
            act = relu_h * relu_h  # ReLU^2
            out = act @ w_proj.t()
            ctx.save_for_backward(x, w_fc, w_proj, h, relu_h)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, w_fc, w_proj, h, relu_h = ctx.saved_tensors
            dtype = grad_output.dtype
            # grad through second linear
            d_act = grad_output @ w_proj.to(dtype)
            # grad through ReLU^2: d/dx[relu(x)^2] = 2*relu(x) * (x > 0)
            d_h = 2 * relu_h.to(dtype) * d_act
            # weight grads
            act_sq = (relu_h * relu_h).to(dtype)
            grad_w_proj = grad_output.t() @ act_sq
            grad_w_fc = d_h.t() @ x.to(dtype)
            grad_x = d_h @ w_fc.to(dtype)
            return grad_x, grad_w_fc, grad_w_proj

    return ReLUSquaredMLP.apply(x, w_fc, w_proj)
```
