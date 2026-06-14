Let me start from what actually costs me money when I pretrain one of these decoder Transformers. The compute is concentrated in two blocks per layer, attention and the feed-forward block, and at the widths people actually run, the feed-forward block is the bigger pile of FLOPs. It is the simplest possible thing: take each token's vector `x` of width `d`, project it up to a wider width — conventionally `4d` — apply some pointwise nonlinearity, and project it back down to `d`. Two dense matmuls with an elementwise squashing in between, `pre = x W1`, `h = act(pre)`, `out = h W2`. I run this on every token of every layer for hundreds of thousands of steps. So anything I can do to this one block — make it reach a given loss in fewer steps, or make each step finish faster — multiplies across the whole run. And total training cost is steps times step-time, so I have to hold both in my head at once: a change that makes each step 10% slower but cuts the steps to a target loss by 30% is a *win*, and a change that keeps the math identical but removes wasted work per step is also a win. Two levers, the activation and the execution. Almost everyone treats both as frozen — ReLU or GELU, executed as a stack of generic library calls — and I want to question both.

Take the activation first. The default is ReLU, `max(x, 0)`: zero below threshold, linear above. GELU and Swish are the fancier defaults, `x Φ(x)` and `x σ(βx)`, but if I look at their large-`x` behavior they are all the same animal — asymptotically linear. For big positive pre-activation they all just pass the input through roughly proportionally. That's a choice, and I should ask whether it's the right one, because there's an old result staring at me from a completely different corner of the field. Krotov and Hopfield, studying associative memory, used polynomial and rectified-polynomial energy terms, `F(s) = s^p` on the positive branch. In their energy notation `p = 2` is the classical Hopfield case, and larger `p` makes each stored pattern's contribution sharper, so the random-memory capacity grows like `N^{p-1}`. But the feed-forward activation that falls out of that energy is one degree lower: the `p = 2` energy case corresponds to ReLU, while `p = 3` corresponds to a rectified parabola. That is the part I need. They left an explicit open question hanging: past the threshold, should the activation grow linearly, sub-linearly, or *faster than linearly* — and might a higher rectified polynomial actually be better than ReLU in ordinary networks? Nobody, as far as I know, has made the next rectified-polynomial activation work as a Transformer activation. The whole pointwise-activation literature answered "grow linearly." Let me take the other branch seriously and ask what the degree-two positive branch does in my FFN.

So the candidate is `act(x) = max(x, 0)²` — rectify, then square. Squared ReLU. Before I get excited let me make sure it isn't obviously broken. Asymptotics: for large positive `x` it grows like `x²`, genuinely faster than linear, which is the whole point of leaving ReLU's degree-one positive branch — it is a different shape from ReLU/GELU/Swish, not a tweak of the same shape. Below zero it is flat zero, the same sparsity-inducing dead zone as ReLU. Numerics: is `x²` going to blow up? In an FFN the pre-activation is the output of a normalized-input linear layer, so it sits at `O(1)`; squaring an `O(1)` number is fine, and crucially I'm *not* tempted to go to cubic or quartic activations, because higher powers amplify the tail viciously — in bf16 a pre-activation of, say, 8 becomes 4096 at degree four, and the gradients scale like `q·x^(q-1)`, so the cube and quartic court overflow and unstable gradients. Degree two is the *minimal* super-linear rectified polynomial activation: it leaves the linear regime by the smallest step, keeping the numerics tame while getting the sharper nonlinearity. That's my first reason to prefer the square specifically and not just "some higher power."

But "sharper basins in a Hopfield net" is an analogy, not a mechanism for why it would help a *language model* FFN. I want a concrete reason. Let me look at what the actually-better Transformer FFNs are doing, because there's a strong recent signal there: the gated variants. The GLU idea (Dauphin) and Shazeer's FFN versions of it replace the first linear-plus-activation with `(act(x W) ⊙ (x V)) W2` — two separate linear projections of the input, one of them activated, multiplied together elementwise, then projected down. ReGLU is the rectified version, `(max(0, x W) ⊙ (x V)) W2`; GEGLU and SwiGLU use GELU and Swish in the gate. These reliably beat the plain ReLU/GELU FFN on held-out perplexity at matched parameters. The standard story for *why* is the multiplicative interaction: the term `(x W) ⊙ (x V)` lets the block compute products of two learned linear features, an input-dependent gate, which a single univariate pointwise activation simply cannot represent — each plain-activation output unit is a fixed function of one pre-activation coordinate, no cross-talk, no products. So the lesson from the gated variants is: *multiplicative interactions are what's buying the quality*.

Now stare at squared ReLU next to ReGLU and something clicks. ReGLU is `max(0, x W) ⊙ (x V)`. What if the two projections are the *same*, `W = V`? Then it's `max(0, x W) ⊙ (x W)`. And `max(0, z) · z` is `z²` when `z > 0` and `0` when `z ≤ 0` — that's exactly `max(0, z)² = relu(z)²`. So squared ReLU applied to `x W` *is* ReGLU with its two gate matrices tied. It is the diagonal, weight-shared special case of the gated unit. Which means squared ReLU is not a different idea from the gates — it is the *same multiplicative interaction*, `(xW) ⊙ (xW)`, the unit gating against itself, the precise mechanism the gated-FFN literature credits for the quality gain. And it gets that interaction for free: ReGLU needs three weight matrices `W, V, W2` and has to shrink the intermediate width by `2/3` to keep the parameter count matched; squared ReLU keeps the original two-matrix FFN, no extra `V`, no `2/3` bookkeeping, no narrower bottleneck. Same self-gating multiplicative effect, fewer parameters, simpler. That's the real reason the square is the sweet spot and not just "Krotov says higher is sharper": it is exactly the power at which `relu(x)·x` collapses into a self-gate, recovering the gated-FFN benefit without the gate's extra matrix. So squared ReLU it is — `h = max(pre, 0)²`, two matrices, the multiplicative interaction baked into the nonlinearity itself.

I do need its gradient before I can train it, and I want it clean. `f(z) = relu(z)² = (max(z, 0))²`. For `z > 0`, `f = z²` and `f'(z) = 2z`, which is `2·relu(z)` since `relu(z) = z` there. For `z < 0`, `f = 0` and `f'(z) = 0`, which is also `2·relu(z)` since `relu(z) = 0` there. At `z = 0` both pieces meet at `0`, and `2·relu(0) = 0`, so there's no jump. So the derivative is just `f'(z) = 2·max(z, 0) = 2·relu(z)`, everywhere, with no case split needed in code. And notice that `f'` is *continuous* — it ramps smoothly through zero — unlike plain ReLU whose derivative jumps from `0` to `1` at the origin. Squared ReLU is `C¹`, a smoother optimization surface than ReLU while keeping the hard zero below threshold. That's a quiet bonus. The activation derivative itself needs only the pre-activation `pre`; for the down-projection weight gradient I will also keep the actual activated tensor `post` that the forward matmul consumed.

Good. I have an activation that should improve quality-per-step. Now the second lever: per-step throughput, and here the activation choice and the execution turn out to be entangled in a way I find satisfying. Let me think about how this block actually runs on a GPU as a stack of library calls, which is what everyone does. First matmul: `pre = x W1ᵀ`, producing the wide `(tokens × 4d)` intermediate — call its size `M × N` with `N = 4d`. cuBLAS does this beautifully, it's compute-bound, lots of FLOPs per byte. Then the activation: a separate elementwise kernel reads all `MN` elements of `pre` from HBM, computes `relu(·)²`, writes all `MN` elements back to HBM. Then the second matmul reads those `MN` elements again. Let me actually count the bytes the way I'd count them for a memory-bound op, because that exposes the waste. The activation kernel does `O(MN)` trivial flops — one max, one multiply per element — for `MN` reads plus `MN` writes. Its arithmetic intensity is at the floor; it is purely *bandwidth*-bound. Its runtime is essentially the time to stream the largest tensor in the block through HBM twice. And that traffic accomplishes nothing the matmul couldn't have done in passing: the matmul already had the value of every `pre` element sitting in a register the instant it finished accumulating it, and instead of using it, the library wrote it to HBM, let the next kernel read it back, transform it, and write it again.

That's the leak. The matmul computes each output tile's accumulator on-chip — in registers, in fp32 — and then, in its epilogue, before it ever touches HBM, it has `pre` right there. If I could run `relu(·)²` *on the accumulator, in registers, before the store*, the activation would cost essentially nothing: no separate kernel launch, no HBM round-trip for `pre`, just a max and a multiply folded into the down-cast I was going to do anyway. The activation's `2·MN` of round-trip HBM traffic — the dominant cost of that bandwidth-bound step — simply vanishes. The roofline picture says this is close to free: the matmul is compute-bound, so the few extra elementwise flops in its epilogue hide under the arithmetic that's already saturating the cores; I'm spending idle compute to erase real memory traffic. This is exactly the kind of move that doesn't change the math at all and just deletes overhead.

The reason nobody does this with the vendor library is that the library won't let me. cuBLAS gives me a matmul with a *fixed* epilogue — down-cast and store — and no hook to staple a deep-learning activation onto the accumulator. The operation I want, "matmul then `relu²` in the same kernel," isn't in its menu, and that's structural, not a tuning knob. So I need to write the matmul myself, at the tile level, where the epilogue is mine. The tile-programming model gives me exactly that: I write the matmul as a blocked accumulation. Each program instance owns one `BLOCK_M × BLOCK_N` tile of the output. It loops over the contraction dimension `K` in chunks of `BLOCK_K`, loading a `BLOCK_M × BLOCK_K` tile of `A` and a `BLOCK_K × BLOCK_N` tile of `B` into on-chip memory and doing `acc += dot(a, b)` into a register-resident accumulator. I keep that accumulator in fp32 for accuracy — standard, because summing many bf16 products loses bits — and I down-cast to bf16 only on the way out, which also halves the store bandwidth. And the whole point: after the `K`-loop finishes, the accumulator holds the full `pre` tile in registers, and *that's where my epilogue runs*.

Let me think about whether the activation is even fusable like this, because not every elementwise-looking op is. Softmax and layernorm, for instance, need a reduction across the row — a max, a sum — so to fuse them into a matmul epilogue you'd need the whole row resident, which crosses tile boundaries. Squared ReLU has no such problem: `relu(z)²` is *strictly local*, one element at a time, no reduction, no neighbor. So it folds into the per-tile epilogue with zero cross-tile communication — `relu(acc)²` on the `BLOCK_M × BLOCK_N` accumulator tile is just an elementwise op on data I already hold. The self-gating activation I chose for quality reasons turns out to be the *easiest possible* thing to fuse for throughput reasons. That's the entanglement I liked: a gated FFN like ReGLU would need me to also compute `x V` — a second matmul or a second projection — and multiply, which is more to fuse and more traffic; squared ReLU's self-gate needs only the one accumulator I already have. The two levers reinforce each other.

So the forward kernel: tile the output, accumulate `pre = x @ W1ᵀ` in fp32, then in the epilogue compute `post = relu(acc)²`, down-cast, store `post`. One subtlety for training: backward needs two different saved values. It needs `pre` for the activation derivative `2·relu(pre)`, and it needs the actual `post` consumed by the second matmul for `∂L/∂W2 = gᵀ @ post`. Reconstructing `post` later from a rounded stashed `pre` would be close, but not exactly the gradient of the mixed-precision forward value that flowed into the down-projection. Cleaner to save both tensors produced by the epilogue: `post` for the next matmul and for `W2`'s gradient, and `pre` for the activation derivative. That adds one extra write of the wide tile for training, but it still deletes the separate activation kernel's read-plus-write round-trip. The important distinction is that the activation is no longer a standalone HBM streaming pass; the stashed `pre` is a deliberate backward cache.

Now the second matmul, `out = post @ W2ᵀ`. There's no activation after it — the residual add and dropout happen outside the block — so there's nothing to fuse into *its* epilogue. It's a plain compute-bound matmul, so I'll just hand it to the regular matmul path (the same tiled kernel without an activation epilogue, or the library); the win was specifically removing the activation round-trip on the *wide* intermediate, which this fusion already did. I don't need to fuse the second matmul into the first either — they're separated by nothing but the activation I already folded in, and keeping them as two kernels lets each be tiled and tuned independently.

Let me get the backward exactly right, because a fused kernel with a wrong gradient is worse than useless. Forward, in matrix form with `x : (M, K)`, `W1 : (N, K)` (so `N = 4d`), `W2 : (d, N)`:

  `pre  = x @ W1ᵀ`        (M, N)
  `post = relu(pre)²`      (M, N)
  `out  = post @ W2ᵀ`      (M, d)

Given the incoming gradient `g = ∂L/∂out`, shape `(M, d)`. Walk it back. Through the second matmul, `out = post @ W2ᵀ`, so `∂L/∂post = g @ W2` (shape `(M, N)`: `(M,d)·(d,N)`), and `∂L/∂W2 = gᵀ @ post` (shape `(d, N)`). Call `d_post = g @ W2`. Through the activation, elementwise, using `f'(pre) = 2·relu(pre)`: `d_pre = 2·relu(pre) ⊙ d_post`, shape `(M, N)`. Through the first matmul, `pre = x @ W1ᵀ`, so `∂L/∂x = d_pre @ W1` (shape `(M, K)`) and `∂L/∂W1 = d_preᵀ @ x` (shape `(N, K)`). Let me sanity-check the activation derivative one more time in this context: the chain through `post = relu(pre)²` multiplies the upstream `d_post` by `dpost/dpre = 2·relu(pre)`, elementwise, yes. And `relu(pre)` is exactly what I stashed, so backward reads `pre`, forms `relu(pre)`, and the rest is matmuls. Good — and notice the symmetry: `d_pre = 2·relu(pre) ⊙ (g @ W2)` is itself a matmul-then-elementwise. The minimal scaffold can compute `g @ W2` as an ordinary matmul and then multiply by `2·relu(pre)`. A tuned implementation can reuse the tiled kernel in a backward mode: feed it `g` and `W2`, read the stashed `pre` as an auxiliary tensor, and change the epilogue from `relu(acc)²` to `2·acc ⊙ relu(pre)`. That is the production shape I want, but the algebra is the same either way.

Let me write the autograd `Function` so it drops into the training loop with the fixed signature `fused_mlp_forward(x, w_fc, w_proj)`. Forward calls the fused kernel to get `pre` and `post`, does the second matmul, and saves `x, w_fc, w_proj, pre, post` for backward. Backward uses the saved `post` for `W2`'s gradient and the saved `pre` for the activation derivative. Everything follows the dtype the autocast context hands me, with the contraction accumulated in fp32 inside the kernel.

Here's the kernel and the wrapper, filling the one empty slot — the body of `fused_mlp_forward` and the gradients it implies:

```python
import torch
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _matmul_relu_sq_kernel(
    a_ptr, b_ptr, c_ptr, pre_ptr,        # a = x, b = W1ᵀ, c = post, pre = stashed pre-activation
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SAVE_PRE: tl.constexpr,              # forward stashes pre for backward
):
    # this program instance owns one BLOCK_M x BLOCK_N tile of the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # block pointers into A (M x K) and B (K x N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # fp32 accumulator lives in registers; sum the contraction in full precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)              # pre tile, accumulated in fp32
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    # epilogue, on the register-resident accumulator, before any HBM store:
    pre = acc.to(tl.bfloat16)                       # the pre-activation, already in hand
    relu_val = tl.maximum(acc, 0.0)                 # relu in fp32
    result = (relu_val * relu_val).to(tl.bfloat16)  # squared ReLU = self-gate, fused

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=mask)             # store post without a separate activation pass
    if SAVE_PRE:                                     # stash pre for backward (cheaper than recompute)
        pre_ptrs = pre_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(pre_ptrs, pre, mask=mask)


class _FusedLinearReLUSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fc, w_proj):
        M, K = x.shape
        N = w_fc.shape[0]                            # N = 4d
        post = torch.empty((M, N), device=x.device, dtype=x.dtype)
        pre = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                             triton.cdiv(N, meta['BLOCK_N']))
        b = w_fc.t().contiguous()                    # B = W1ᵀ, shape (K, N)
        _matmul_relu_sq_kernel[grid](
            x, b, post, pre,
            M, N, K,
            x.stride(0), x.stride(1),
            b.stride(0), b.stride(1),
            post.stride(0), post.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            SAVE_PRE=True,
        )
        out = post @ w_proj.t()                      # second matmul; nothing to fuse after it
        ctx.save_for_backward(x, w_fc, w_proj, pre, post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, w_fc, w_proj, pre, post = ctx.saved_tensors
        dtype = grad_output.dtype
        d_post = grad_output @ w_proj.to(dtype)                          # g @ W2  -> (M, N)
        grad_w_proj = grad_output.reshape(-1, grad_output.shape[-1]).t() @ \
                      post.to(dtype).reshape(-1, post.shape[-1])              # gᵀ @ post -> (d, N)
        d_pre = 2 * F.relu(pre).to(dtype) * d_post                       # 2·relu(pre) ⊙ d_post
        grad_x = d_pre @ w_fc.to(dtype)                                  # d_pre @ W1 -> (M, K)
        grad_w_fc = d_pre.reshape(-1, d_pre.shape[-1]).t() @ x.to(dtype).reshape(-1, x.shape[-1])  # (N, K)
        return grad_x, grad_w_fc, grad_w_proj


def fused_mlp_forward(x, w_fc, w_proj):
    """FFN forward: up-project, squared-ReLU (fused into the matmul epilogue), down-project."""
    return _FusedLinearReLUSquare.apply(x, w_fc, w_proj)
```

Let me trace the causal chain one more time to be sure it hangs together. I started with the FFN dominating my training cost and two levers I'd been told to leave alone. On the activation, I took Krotov–Hopfield's open question seriously — past the threshold, grow faster than linear — and landed on the minimal super-linear rectified-polynomial activation, squared ReLU; the square rather than higher powers because higher powers wreck bf16 numerics and gradients, and because exactly at degree two the unit `relu(x)·x` becomes a self-gate, `relu(xW) ⊙ (xW)`, which is ReGLU with tied weights. So squared ReLU delivers the multiplicative interaction that the gated FFNs credit for their quality gains, but for free: two matrices instead of three, no `2/3` width shrink, and a `C¹` derivative `2·relu(z)` that's cleaner than ReLU's. On the execution, I counted the bytes and saw the activation is a purely bandwidth-bound elementwise pass that round-trips the wide `(tokens × 4d)` intermediate through HBM, doing nothing the matmul couldn't have done while the accumulator was still in registers — and the vendor library can't fold it in because its epilogue is fixed. So I wrote the matmul myself at the tile level, accumulating `pre` in an fp32 register accumulator, and ran `relu(·)²` in the epilogue before the single down-cast-and-store, which deletes the activation's standalone round-trip at near-zero cost because the matmul is compute-bound and hides the extra flops. Squared ReLU being strictly local, no reduction, is precisely what makes it trivially fusable — the quality lever and the throughput lever chose the same activation. I save `pre` for the derivative and `post` for the down-projection gradient; a production kernel can fuse the backward `g @ W2` matmul with the `2·relu(pre)` epilogue as well. The result drops into the existing FFN slot with the fixed signature, the squared-ReLU MLP algebra, fewer HBM bytes per step, and a better activation per step.
