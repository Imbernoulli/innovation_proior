# Fused squared-ReLU FFN (Triton), distilled

The Transformer feed-forward block — up-project to `4d`, apply a nonlinearity, project back
to `d` — is improved on two axes at once. (1) **Activation:** replace ReLU/GELU with
**squared ReLU**, `act(x) = max(x, 0)²`. (2) **Execution:** fuse the up-projection matmul and
the squared-ReLU activation into a single tiled GPU (Triton) kernel, applying the activation
on the fp32 accumulator in registers so the wide intermediate never makes a round-trip to HBM,
with a custom autograd backward. Same math as a squared-ReLU MLP; fewer memory bytes per step
and better quality per step.

## Problem it solves

Reduce the training cost (steps × step-time) of an auto-regressive Transformer LM by improving
the FFN block — a better activation that lowers validation loss in fewer steps, and an
execution that removes the activation's wasted HBM traffic — while keeping the math and
gradients exactly correct so it drops into an existing training loop unchanged.

## Key idea 1 — squared ReLU

`act(z) = max(z, 0)² = relu(z)²`.

- **Super-linear rectified polynomial.** Krotov & Hopfield motivate the question of whether a
  rectified unit should grow faster than linearly on its positive branch. In their energy notation,
  power `p = 2` corresponds to ReLU as a feed-forward activation, and the next case corresponds to
  the rectified parabola. Squared ReLU is that minimal super-linear activation: unlike ReLU, GELU,
  and Swish, which are asymptotically linear, it grows like `z²` for large `z`. Higher activation
  powers blow up bf16 activations and gradients (`grad ∝ q·z^(q-1)`), so the square is the smallest
  practical step past ReLU.
- **Self-gating = free multiplicative interaction.** `max(0, z)·z = relu(z)²`, so squared ReLU
  applied to `xW` equals `max(0, xW) ⊙ (xW)`, which is **ReGLU with tied weights** (ReGLU is
  `(max(0, xW) ⊙ (xV)) W2`, set `V = W`). It captures the multiplicative interaction that GLU
  variants (ReGLU/GEGLU/SwiGLU, Shazeer 2020) credit for their quality gains, but with **two**
  weight matrices instead of three — no extra `V`, no `2/3` intermediate-width shrink.
- **`C¹` derivative.** `d/dz relu(z)² = 2·max(z, 0) = 2·relu(z)` everywhere (continuous through
  the origin, unlike ReLU's jump). The activation derivative needs only `pre`; the full backward
  also saves `post` for the down-projection weight gradient.

## Key idea 2 — fuse the activation into the matmul epilogue

The FFN run as separate library ops materializes the wide `(M, N)` intermediate (`N = 4d`) to
HBM, then a bandwidth-bound activation kernel reads `MN`, computes `relu(·)²`, writes `MN` — a
full round-trip of the block's largest tensor doing trivial arithmetic. A tiled matmul already
holds each output tile's `pre` in an fp32 register accumulator after the K-loop; applying
`relu(·)²` there, before the store of the activation output, deletes that separate `2·MN` HBM
traffic. Squared ReLU is strictly local (no reduction), so it fuses into the per-tile epilogue with
no cross-tile communication. The vendor matmul library cannot do this (fixed epilogue), so the
matmul is written at the tile level. In training, save both `pre` (for `2·relu(pre)`) and `post`
(for `grad_w_proj = gᵀ @ post`); this is still cheaper than a standalone activation read/write
pass and keeps the backward faithful to the mixed-precision forward value.

## Forward / backward (matrix form)

`x : (M, K)`, `w_fc = W1 : (N, K)` with `N = 4d`, `w_proj = W2 : (d, N)`:

```
pre  = x @ W1ᵀ                 # (M, N)   accumulated fp32, activation fused in epilogue
post = relu(pre)²              # (M, N)
out  = post @ W2ᵀ              # (M, d)   plain matmul (nothing to fuse after it)
```

Given `g = ∂L/∂out` (M, d):

```
d_post      = g @ W2                   # (M, N)
grad_w_proj = gᵀ @ post                # (d, N)
d_pre       = 2·relu(pre) ⊙ d_post     # (M, N)   uses f'(pre) = 2·relu(pre)
grad_x      = d_pre @ W1               # (M, K)
grad_w_fc   = d_preᵀ @ x               # (N, K)
```

(The backward `d_pre = 2·relu(pre) ⊙ (g @ W2)` is itself matmul-then-elementwise, so it can
reuse the same fused kernel in a backward mode that reads the stashed `pre`.)

## Working code

Fills the `fused_mlp_forward(x, w_fc, w_proj)` slot of the FFN. Forward: a Triton tiled matmul
`x @ W1ᵀ` with the squared-ReLU activation fused into the accumulator epilogue (saving `pre` and
`post`), then the down-projection. Backward: a custom autograd `Function` implementing the
gradients above.

```python
import torch
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _matmul_relu_sq_kernel(
    a_ptr, b_ptr, c_ptr, pre_ptr,                 # a = x, b = W1ᵀ, c = post, pre = pre-activation
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # fp32 register accumulator
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    # epilogue on the accumulator, before any HBM store:
    pre = acc.to(tl.bfloat16)
    relu_val = tl.maximum(acc, 0.0)                        # relu in fp32
    result = (relu_val * relu_val).to(tl.bfloat16)         # squared ReLU (self-gate), fused

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=mask)                    # store post; no separate activation pass
    if SAVE_PRE:
        pre_ptrs = pre_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(pre_ptrs, pre, mask=mask)                 # stash pre for backward


class _FusedLinearReLUSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        M, K = x.shape
        N = w_fc.shape[0]                                  # N = 4d
        post = torch.empty((M, N), device=x.device, dtype=x.dtype)
        pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        b = w_fc.t().contiguous()                          # B = W1ᵀ  (K, N)
        _matmul_relu_sq_kernel[grid](
            x, b, post, pre,
            M, N, K,
            x.stride(0), x.stride(1),
            b.stride(0), b.stride(1),
            post.stride(0), post.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            SAVE_PRE=True,
        )
        out = post @ w_proj.t()                            # down-projection (plain matmul)
        ctx.save_for_backward(x, w_fc, w_proj, pre, post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, pre, post = ctx.saved_tensors
        dtype = grad_output.dtype
        d_post = grad_output @ w_proj.to(dtype)                          # g @ W2
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                      post.to(dtype).reshape(-1, post.shape[-1])              # gᵀ @ post
        d_pre = 2 * F.relu(pre).to(dtype) * d_post                       # 2·relu(pre) ⊙ d_post
        grad_x = d_pre @ w_fc.to(dtype)                                  # d_pre @ W1
        grad_w_fc = d_pre.reshape(-1, d_pre.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])
        return grad_x, grad_w_fc, grad_w_proj


def fused_mlp_forward(x, w_fc, w_proj):
    """FFN forward with a Triton fused linear + squared-ReLU kernel."""
    return _FusedLinearReLUSquare.apply(x, w_fc, w_proj)
```

A pure-PyTorch squared-ReLU variant (no Triton, custom autograd, same math — useful as a
reference and on hardware without Triton):

```python
def relu_sq_mlp_forward(x, w_fc, w_proj):
    class ReLUSquaredMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_fc, w_proj):
            pre = x @ w_fc.t()
            relu_pre = F.relu(pre)
            post = relu_pre * relu_pre          # squared ReLU
            out = post @ w_proj.t()
            ctx.save_for_backward(x, w_fc, w_proj, relu_pre, post)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, w_fc, w_proj, relu_pre, post = ctx.saved_tensors
            dtype = grad_output.dtype
            d_post = grad_output @ w_proj.to(dtype)
            d_pre = 2 * relu_pre.to(dtype) * d_post          # f'(pre) = 2·relu(pre)
            grad_w_proj = grad_output.t() @ post.to(dtype)
            grad_w_fc = d_pre.t() @ x.to(dtype)
            grad_x = d_pre @ w_fc.to(dtype)
            return grad_x, grad_w_fc, grad_w_proj

    return ReLUSquaredMLP.apply(x, w_fc, w_proj)
```

## Production-grade kernel notes

A fully optimized version (e.g. the modded-nanogpt implementation) adds, on top of this core:
persistent tiling across the GPU's SMs, tuned `BLOCK_M/N/K` and `num_warps`/`num_stages`,
tensor-descriptor loads, and a backward mode of the same fused kernel. In that production layout
the down-projection weight is stored transposed relative to the `nn.Linear` scaffold (`post @ W2`
instead of `post @ w_proj.t()`), forward returns both `pre` and `post`, `post` is saved for
`dW2 = postᵀ @ g`, and the backward-mode kernel computes
`d_pre = 2·(g @ W2)·relu(pre)` by reading `pre` as an auxiliary tensor.
