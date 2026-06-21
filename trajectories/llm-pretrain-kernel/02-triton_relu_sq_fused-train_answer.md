The GELU floor landed exactly where I expected on quality and told me something sharp on throughput. It posted `val_loss` 2.2868, `wikitext2_ppl` 43.94, `lambada_ppl` 68.2 — the unshaped-GELU level, the price of an asymptotically-linear activation that pushes a strongly-firing and a barely-firing unit through at nearly the same slope and adds no super-linear shaping. And it posted `elapsed` 20035 by leaving the matmuls to cuBLAS, confirming that the two matmuls plus the *single HBM round-trip of the wide hidden* dominate — tidying the activation's elementwise chain into one launch was respectable, not transformative, because I deliberately left the matmuls in torch and so left that round-trip on the table. So the floor handed me two distinct openings, one per axis, and this rung goes after both at once: change the activation to something with real super-linear shaping, and execute it by folding it into the matmul that produces it so the leftover round-trip disappears too.

The activation leads because the quality story has to. ReLU, GELU, and Swish are all the same animal in the tail — asymptotically linear — so they differ only near the origin, the soft knee, which is why GELU historically barely beat ReLU and why 2.2868 sat at the unshaped level. Reshaping the knee is deck-chairs; the real lever is the *asymptotics*. So I propose **squared ReLU**, $\mathrm{act}(z) = \max(z,0)^2$ — rectify, then square. For large positive $z$ it grows like $z^2$, genuinely faster than linear, a *different shape* from GELU rather than a tweak of the same shape; below zero it is flat zero, the same sparsity-inducing dead zone as ReLU. The numerics stay tame because in this FFN the pre-activation is the output of a normalized-input linear layer under bf16 autocast, so it sits at $O(1)$, and squaring an $O(1)$ number is fine. I am explicitly *not* tempted to go cubic or quartic: higher powers amplify the tail viciously — a pre-activation of 8 becomes 4096 at degree four in bf16 — and the gradient of a degree-$q$ rectified power scales like $q\cdot z^{q-1}$, so cube and quartic court overflow and unstable gradients. Degree two is the *minimal* super-linear rectified polynomial: it leaves the linear regime by the smallest step, keeping the numerics tame while getting the sharper nonlinearity.

That "minimal super-linear" framing has a precedent worth naming. Krotov and Hopfield, studying associative-memory capacity, used rectified-polynomial energy terms $F(s)=s^p$ on the positive branch and left an explicit open question hanging — past threshold, should an activation grow linearly, sub-linearly, or *faster than linearly*, and might a higher rectified polynomial beat ReLU in ordinary networks? The whole pointwise literature answered "linearly," and the GELU floor is that answer's price tag. But "sharper basins in a Hopfield net" is an analogy, not a mechanism for a *language model* FFN, and the concrete mechanism comes from the gated FFNs. The GLU idea (Dauphin et al.) and Shazeer's variants replace the first linear-plus-activation with $(\mathrm{act}(xW) \odot (xV))\,W_2$ — two separate linear projections of the input, one squashed, multiplied elementwise — and they reliably beat plain ReLU/GELU on held-out perplexity at matched parameters. The standard reason *why* is the multiplicative interaction: $(xW)\odot(xV)$ computes products of two learned linear features, an input-dependent gate, which a single univariate pointwise activation cannot represent — each plain-activation output is a fixed function of one pre-activation coordinate, no cross-talk, no products. The lesson is that *multiplicative interactions are what buy the quality*. The catch — the reason this whole ladder lives in a two-matrix slot — is that the GLU variants need a third weight matrix $V$ and a $2/3$ inner-width shrink to stay parameter-matched, and `fused_mlp_forward` is handed only `w_fc` and `w_proj`. There is no $V$ to pass; a gated FFN cannot be expressed here.

Now squared ReLU next to ReGLU and it clicks. ReGLU is $\max(0,xW)\odot(xV)$; tie the two projections, $W=V$, and it becomes $\max(0,xW)\odot(xW)$. But $\max(0,z)\cdot z$ is $z^2$ when $z>0$ and $0$ when $z\le0$ — exactly $\max(0,z)^2 = \mathrm{relu}(z)^2$. So squared ReLU applied to $xW$ *is* **ReGLU with its two gate matrices tied**, the diagonal weight-shared special case of the gated unit. It is not a different idea from the gates; it is the *same multiplicative interaction* $(xW)\odot(xW)$, the unit gating against itself — the precise mechanism the gated-FFN literature credits for the gain — and it gets that interaction *for free in exactly the slot I am allowed to edit*, keeping the original `w_fc, w_proj` with no extra $V$, no bookkeeping, no narrower bottleneck. The constraint that blocks GLU here is *precisely* what makes squared ReLU the right move. Its gradient is clean and, as a bonus, smooth: $f(z)=\mathrm{relu}(z)^2$ has $f'(z)=2z=2\,\mathrm{relu}(z)$ for $z>0$ and $f'(z)=0=2\,\mathrm{relu}(z)$ for $z<0$, meeting at $0$ with no jump, so $f'(z)=2\,\max(z,0)=2\,\mathrm{relu}(z)$ everywhere — no case split in code, and *continuous* through the origin, making squared ReLU $C^1$, a smoother optimization surface than ReLU while keeping the hard zero below threshold. The activation derivative needs only the pre-activation; the down-projection weight gradient $\partial L/\partial W_2 = g^\top\!@\,\mathrm{post}$ needs the activated tensor.

The throughput lever pays off the floor's leftover round-trip, and here the activation and the execution turn out entangled. The GELU floor left the activation as a standalone HBM pass over the wide $(M,N)$ hidden, $N=4\cdot\texttt{n\_embd}$: the matmul wrote `pre` out, the activation read $MN$ and wrote $MN$, the second matmul read it again. But the matmul already had every `pre` element sitting in an fp32 register accumulator the instant it finished that output tile — and cuBLAS wrote it to HBM instead of using it. If I run the activation *on the accumulator, in registers, before the store*, that $2\cdot MN$ of round-trip traffic — the dominant cost of the bandwidth-bound step — simply vanishes. By the roofline picture it is close to free: the matmul is compute-bound, so the few extra elementwise flops in its epilogue hide under arithmetic already saturating the cores; I spend idle compute to erase real memory traffic. The reason the GELU rung did not do this is structural, not a tuning knob — cuBLAS gives a *fixed* epilogue (down-cast and store) with no hook to staple an activation onto the accumulator — so to fuse I must write the up-projection matmul myself at the tile level, where the epilogue is mine.

And the entanglement pays off: squared ReLU is the *easiest possible* activation to fuse. Softmax or layernorm need a row reduction, so fusing them would require the whole row resident, crossing tile boundaries. $\mathrm{relu}(z)^2$ is strictly local — one element at a time, no reduction, no neighbor — so it folds into the per-tile epilogue with zero cross-tile communication: `relu(acc)²` on the $\texttt{BLOCK\_M}\times\texttt{BLOCK\_N}$ accumulator is just an elementwise op on data I already hold. A gated ReGLU would also need $xV$ computed and multiplied, more to fuse and more traffic; the self-gate needs only the one accumulator. The quality lever and the throughput lever chose the same activation. So the forward kernel tiles the up-projection: each program instance owns one $\texttt{BLOCK\_M}\times\texttt{BLOCK\_N}$ output tile, loops over the contraction $K$ in chunks of `BLOCK_K`, loading sub-tiles of $x$ and `w_fc.t()` and accumulating `acc += dot(a, b)` into an fp32 register accumulator (standard — summing many bf16 products loses bits), and after the K-loop the epilogue computes `relu(acc)²`, down-casts to bf16, and stores `post`. I use $\texttt{BLOCK\_M}=\texttt{BLOCK\_N}=64$, $\texttt{BLOCK\_K}=32$, grid $(\mathrm{cdiv}(M,\texttt{BLOCK\_M}),\,\mathrm{cdiv}(N,\texttt{BLOCK\_N}))$. The second matmul `out = post @ w_proj.t()` has no activation after it (residual and dropout live outside the block), so there is nothing to fuse into its epilogue — I leave it a plain torch matmul, the same decision the GELU rung made for both matmuls, now made for just the down one.

The backward I keep minimal and let the algebra dictate what to stash. With $x:(M,K)$, $w_{fc}:(N,K)$, $w_{proj}:(d,N)$, forward is $\mathrm{pre}=x\,w_{fc}^\top$, $\mathrm{post}=\mathrm{relu}(\mathrm{pre})^2$, $\mathrm{out}=\mathrm{post}\,w_{proj}^\top$. Given $g=\partial L/\partial\mathrm{out}$ of shape $(M,d)$: through the second matmul, $d_{\mathrm{post}}=g\,w_{proj}$ shape $(M,N)$ and $\partial L/\partial w_{proj}=g^\top\mathrm{post}$ shape $(d,N)$; through the activation, $d_{\mathrm{pre}}=2\,\mathrm{relu}(\mathrm{pre})\odot d_{\mathrm{post}}$; through the first matmul, $\partial L/\partial x=d_{\mathrm{pre}}\,w_{fc}$ and $\partial L/\partial w_{fc}=d_{\mathrm{pre}}^\top x$. The only nonlinear-specific value the backward needs is $\mathrm{relu}(\mathrm{pre})$, so I save just `pre` and recompute `relu(pre)` and `relu(pre)²` from it — `relu(pre)²` for the $w_{proj}$ gradient, $2\,\mathrm{relu}(\mathrm{pre})$ for the activation gradient. Saving only `pre` and recomputing, rather than also stashing `post`, trades a cheap elementwise recompute for one fewer wide tensor held across the step. I wrap it in a `torch.autograd.Function` with the fixed signature, accumulating in fp32 inside the kernel and following the autocast dtype outside.

The bet, stated falsifiably against the GELU floor on both axes: on quality I expect a real drop below 2.2868, since squared ReLU is the self-gating multiplicative interaction the GLU variants credit for beating GELU, captured in this exact two-matrix slot — if `val_loss`, `wikitext2_ppl`, and `lambada_ppl` do not all come in under the floor, the "multiplicative interaction buys quality" story is wrong *here* and I should reconsider the activation, not the kernel. On throughput the bet is riskier and I want to be honest: the fusion deletes the $2\cdot MN$ round-trip, but it does so by replacing cuBLAS's exquisitely-tuned up-projection with a hand-rolled tiled matmul at fixed $\texttt{BLOCK\_M}=\texttt{BLOCK\_N}=64$, $\texttt{BLOCK\_K}=32$, and under `torch.compile` the floor's matmuls were already heavily optimized. So the real question this rung answers is whether the round-trip I save is worth more than the cuBLAS-vs-naive matmul gap I take on. If `elapsed` lands at or below 20035, the fusion paid; if it lands *above* 20035 even though quality improved, that is the diagnosis that the cleanest win keeps the matmuls in cuBLAS and changes only the activation — i.e. the next rung should drop the Triton matmul entirely and run squared ReLU in plain torch, betting that almost all the quality gain here came from the activation and almost none from the fusion.

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
