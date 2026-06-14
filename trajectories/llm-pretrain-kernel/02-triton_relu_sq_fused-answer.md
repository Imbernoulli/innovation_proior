**Problem.** The GELU floor (`val_loss` 2.2868, `elapsed` 20035) exposed two openings: GELU is
asymptotically linear, so the activation adds no super-linear shaping (quality lever), and even with the
elementwise chain fused, the activation is still a standalone HBM round-trip of the wide hidden because
the matmuls were left to cuBLAS (throughput lever). Go after both.

**Key idea 1 — squared ReLU.** `act(z) = max(z, 0)²`. The minimal super-linear rectified polynomial
(higher powers blow up bf16 tails; gradient `∝ q·z^{q-1}`). Crucially, `max(0, z)·z = relu(z)²`, so
squared ReLU on `xW` equals `max(0, xW) ⊙ (xW)` — **ReGLU with tied weights** (`V = W`). It captures the
multiplicative interaction GLU variants credit for beating GELU, but with the **two** matrices the edit
surface gives (`w_fc, w_proj`), no third gate matrix `V`, no `2/3` width shrink. Its derivative is `C¹`:
`d/dz relu(z)² = 2·relu(z)` everywhere.

**Key idea 2 — fuse the activation into the up-matmul epilogue.** A tiled Triton matmul holds each
output tile's `pre` in an fp32 register accumulator after the K-loop; running `relu(acc)²` there, before
the store, deletes the activation's `2·MN` HBM round-trip (cuBLAS can't, its epilogue is fixed). Squared
ReLU is strictly local (no reduction), so it fuses with zero cross-tile communication. The down-matmul
has nothing after it, so it stays a plain torch matmul.

**Why.** Quality should drop below 2.2868 (the self-gating interaction); the fusion deletes the
round-trip. The risk: it replaces cuBLAS's up-projection with a hand-rolled tiled matmul (fixed tiles),
which `torch.compile` already optimized — so `elapsed` may *rise* if the cuBLAS-vs-naive gap exceeds the
round-trip saved.

**Backward.** Save only `pre`; recompute `relu(pre)²` for `∂L/∂w_proj` and `2·relu(pre)` for the
activation grad (one fewer wide tensor held than stashing `post`). The two linears' grads are matmuls.

**Hyperparameters.** `BLOCK_M = BLOCK_N = 64`, `BLOCK_K = 32`; fp32 accumulator, bf16 store; grid
`(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`. No `CONFIG_OVERRIDES`.

```python
# EDITABLE region of custom_pretrain.py (lines 33-48) — step 2: Triton fused linear + ReLU^2
import triton
import triton.language as tl

@triton.jit
def _matmul_relu_sq_kernel(
    a_ptr, b_ptr, c_ptr, pre_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SAVE_PRE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K
    pre = acc.to(tl.bfloat16)
    relu_val = tl.maximum(acc, 0.0)
    result = (relu_val * relu_val).to(tl.bfloat16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=mask)
    if SAVE_PRE:
        pre_ptrs = pre_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(pre_ptrs, pre, mask=mask)

class _FusedLinearReLUSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        M, K = x.shape
        N = w_fc.shape[0]
        post = torch.empty((M, N), device=x.device, dtype=x.dtype)
        pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        b = w_fc.t().contiguous()
        _matmul_relu_sq_kernel[grid](
            x, b, post, pre,
            M, N, K,
            x.stride(0), x.stride(1),
            b.stride(0), b.stride(1),
            post.stride(0), post.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            SAVE_PRE=True,
        )
        out = post @ w_proj.t()
        ctx.save_for_backward(x, w_fc, w_proj, pre)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, pre = ctx.saved_tensors
        dtype = grad_output.dtype
        d_post = grad_output @ w_proj.to(dtype)
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                      F.relu(pre).pow(2).to(dtype).reshape(-1, pre.shape[-1])
        d_pre = 2 * F.relu(pre).to(dtype) * d_post
        grad_x = d_pre @ w_fc.to(dtype)
        grad_w_fc = d_pre.reshape(-1, d_pre.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
        return grad_x, grad_w_fc, grad_w_proj

def fused_mlp_forward(x, w_fc, w_proj):
    """MLP forward with Triton fused linear+ReLU^2 kernel."""
    return _FusedLinearReLUSquare.apply(x, w_fc, w_proj)
```
