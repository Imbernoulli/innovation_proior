The GELU floor landed where I expected on one axis and told me something sharp on the other. On
quality it is unremarkable by construction: `val_loss` 2.2868, `wikitext2_ppl` 43.94, `lambada_ppl`
68.2 — exactly the "unshaped GELU" level I predicted, the floor against which an *activation* change
should move. On throughput it posted `elapsed` 20035 leaving the matmuls to cuBLAS, which confirms the
other half of the prediction: the two matmuls plus the single HBM round-trip of the wide hidden
dominate, and tidying the activation's elementwise chain into one launch was respectable but not
transformative. So the GELU rung handed me two distinct openings. First, the activation is the quality
lever — GELU is asymptotically linear, a strongly-firing and a barely-firing unit pass through at
nearly the same slope, no super-linear shaping — and on LM perplexity that is famously only on par with
ReLU, so 2.2868 is what "no shaping beyond the knee" buys. Second, the activation is *still* a separate
HBM round-trip even after I fused its elementwise chain, because I deliberately left the matmuls in
torch. This rung goes after *both* at once: change the activation to something with real super-linear
shaping, and execute it by folding it into the matmul that produces it, so the round-trip the GELU rung
left on the table disappears too.

Take the activation first, because the quality story has to lead. The defaults — ReLU, GELU, Swish —
are all, in their large-`x` behavior, the same animal: asymptotically linear. For big positive
pre-activation they pass the input through roughly proportionally. That is a *choice*, and the GELU
number says it is a choice that costs me — 2.2868 is the price of asymptotic linearity. There is an old
result staring at me from a different corner: Krotov and Hopfield, studying associative memory, used
rectified-polynomial energy terms `F(s) = s^p` on the positive branch, and noted that the
feed-forward activation falling out of that energy is one degree lower — the `p = 2` energy case
corresponds to ReLU, the `p = 3` case to a rectified parabola — and they left an explicit open question
hanging: past the threshold, should the activation grow linearly, sub-linearly, or *faster than
linearly*, and might a higher rectified polynomial beat ReLU in ordinary networks? The whole pointwise
literature answered "linearly." The GELU floor I just measured is that answer's price tag. Let me take
the other branch.

So the candidate is `act(z) = max(z, 0)²` — rectify, then square. Squared ReLU. Before getting
excited, check it is not obviously broken. Asymptotics: for large positive `z` it grows like `z²`,
genuinely faster than linear — a *different shape* from GELU, not a tweak of the same shape, which is
exactly what I need given that tweaking the knee is what GELU already did relative to ReLU and it bought
nothing on perplexity. Below zero it is flat zero, the same sparsity-inducing dead zone as ReLU.
Numerics: will `z²` blow up? In this FFN the pre-activation is the output of a normalized-input linear
layer under bf16 autocast, so it sits at `O(1)`; squaring an `O(1)` number is fine. And crucially I am
*not* tempted to go cubic or quartic, because higher powers amplify the tail viciously — in bf16 a
pre-activation of 8 becomes 4096 at degree four, and the gradient scales like `q·z^(q−1)`, so cube and
quartic court overflow and unstable gradients. Degree two is the *minimal* super-linear rectified
polynomial: it leaves the linear regime by the smallest step, keeping the numerics tame while getting
the sharper nonlinearity. That is the first reason to pick the square specifically and not just "some
higher power."

But "sharper basins in a Hopfield net" is an analogy, not a mechanism for why it helps a *language
model* FFN. I want a concrete reason, and the gated FFNs supply it. The GLU idea (Dauphin) and
Shazeer's FFN variants replace the first linear-plus-activation with `(act(xW) ⊙ (xV)) W2` — two
separate linear projections of the input, one activated, multiplied elementwise, then projected down.
ReGLU is the rectified version, `(max(0, xW) ⊙ (xV)) W2`. These reliably beat the plain ReLU/GELU FFN
on held-out perplexity at matched parameters, and the standard story for *why* is the multiplicative
interaction: `(xW) ⊙ (xV)` lets the block compute products of two learned linear features, an
input-dependent gate, which a single univariate pointwise activation cannot represent — each
plain-activation output is a fixed function of one pre-activation coordinate, no cross-talk, no
products. So the lesson from the gated variants is that *multiplicative interactions are what buy the
quality*. The catch, and it is the reason this whole ladder lives in a two-matrix slot: the GLU
variants need a *third* weight matrix `V` and shrink the inner width by `2/3` to stay
parameter-matched, and the edit surface hands `fused_mlp_forward` only `w_fc` and `w_proj` — there is
no `V` to pass. A gated FFN simply cannot be expressed here.

Now stare at squared ReLU next to ReGLU and it clicks. ReGLU is `max(0, xW) ⊙ (xV)`. Set the two
projections equal, `W = V`: it becomes `max(0, xW) ⊙ (xW)`. And `max(0, z)·z` is `z²` when `z > 0` and
`0` when `z ≤ 0` — exactly `max(0, z)² = relu(z)²`. So squared ReLU applied to `xW` *is* ReGLU with its
two gate matrices tied. It is the diagonal, weight-shared special case of the gated unit, which means
it is not a different idea from the gates — it is the *same multiplicative interaction*, `(xW) ⊙ (xW)`,
the unit gating against itself, the precise mechanism the gated-FFN literature credits for the gain.
And it gets that interaction *for free in exactly the slot I am allowed to edit*: ReGLU needs `W, V, W2`
and the `2/3` shrink; squared ReLU keeps the original two matrices `w_fc, w_proj`, no extra `V`, no
bookkeeping, no narrower bottleneck. So the constraint that blocks GLU here is *precisely* what makes
squared ReLU the right move: it recovers the gated benefit without the gate's third matrix. That is the
real reason the square is the sweet spot and not just "Krotov says higher is sharper" — it is exactly
the power at which `relu(z)·z` collapses into a self-gate.

I need the gradient before I can train it, and I want it clean. `f(z) = relu(z)² = (max(z,0))²`. For
`z > 0`, `f = z²`, `f'(z) = 2z = 2·relu(z)` since `relu(z) = z` there. For `z < 0`, `f = 0`,
`f'(z) = 0 = 2·relu(z)` since `relu(z) = 0` there. At `z = 0` both pieces meet at `0`, and
`2·relu(0) = 0`, no jump. So the derivative is just `f'(z) = 2·max(z, 0) = 2·relu(z)` everywhere, no
case split in code — and notice it is *continuous*, ramping smoothly through zero, unlike ReLU's
derivative which jumps 0→1 at the origin. Squared ReLU is `C¹`, a smoother optimization surface than
ReLU while keeping the hard zero below threshold. A quiet bonus. The activation derivative needs only
the pre-activation; the down-projection weight gradient `∂L/∂W2 = gᵀ @ post` needs the activated tensor
`post`.

Now the throughput lever, and this is where the GELU rung's leftover round-trip gets paid off and where
the activation choice and the execution turn out entangled. The GELU floor left the activation as a
standalone HBM pass over the wide `(M, N)` hidden, `N = 4·n_embd`: the matmul wrote `pre` out, the
activation read `MN`, wrote `MN`, the second matmul read it again. But the matmul already had every
`pre` element sitting in an fp32 register accumulator the instant it finished accumulating that output
tile — and instead of using it, cuBLAS wrote it to HBM. If I run the activation *on the accumulator, in
registers, before the store*, that `2·MN` of round-trip traffic — the dominant cost of the
bandwidth-bound activation step — simply vanishes. By the roofline picture it is close to free: the
matmul is compute-bound, so the few extra elementwise flops in its epilogue hide under arithmetic that
is already saturating the cores; I spend idle compute to erase real memory traffic. The reason the GELU
rung did not do this is that cuBLAS gives a *fixed* epilogue (down-cast and store) with no hook to
staple an activation onto the accumulator — it is structural, not a tuning knob — so to fuse I have to
write the up-projection matmul myself at the tile level, where the epilogue is mine.

And here the entanglement pays: squared ReLU is the *easiest possible* activation to fuse. Softmax or
layernorm need a row reduction, so fusing them into a matmul epilogue would need the whole row resident,
crossing tile boundaries. `relu(z)²` is strictly local — one element at a time, no reduction, no
neighbor — so it folds into the per-tile epilogue with zero cross-tile communication: `relu(acc)²` on
the `BLOCK_M × BLOCK_N` accumulator is just an elementwise op on data I already hold. A gated unit like
ReGLU would also need `xV` computed and multiplied, more to fuse and more traffic; the self-gate needs
only the one accumulator. The quality lever and the throughput lever chose the same activation.

So the forward kernel tiles the up-projection: each program instance owns one `BLOCK_M × BLOCK_N`
output tile, loops over the contraction `K` in chunks of `BLOCK_K`, loading sub-tiles of `x` and
`w_fc.t()`, accumulating `acc += dot(a, b)` into an fp32 register accumulator (standard — summing many
bf16 products loses bits), and after the K-loop the epilogue computes `relu(acc)²`, down-casts to bf16,
stores `post`. The second matmul `out = post @ w_proj.t()` has no activation after it (residual and
dropout are outside the block), so there is nothing to fuse into its epilogue — I leave it as a plain
torch matmul, the same decision the GELU rung made for *both* matmuls, now made for just the down one.

For backward I keep it minimal and let the algebra dictate what to stash. Forward, with `x : (M, K)`,
`w_fc : (N, K)` so `N = 4·n_embd`, `w_proj : (d, N)`: `pre = x @ w_fc.t()`, `post = relu(pre)²`,
`out = post @ w_proj.t()`. Given `g = ∂L/∂out` of shape `(M, d)`: through the second matmul,
`d_post = g @ w_proj` shape `(M, N)`, and `∂L/∂w_proj = gᵀ @ post` shape `(d, N)`; through the
activation, `d_pre = 2·relu(pre) ⊙ d_post`; through the first matmul, `∂L/∂x = d_pre @ w_fc` and
`∂L/∂w_fc = d_preᵀ @ x`. The only nonlinear-specific value the backward needs is `relu(pre)`, so I save
just `pre` and recompute `relu(pre)` and `relu(pre)²` from it in the backward — `relu(pre)²` for the
`w_proj` gradient and `2·relu(pre)` for the activation gradient. Saving only `pre` and recomputing,
rather than also stashing `post`, trades a cheap elementwise recompute for one fewer wide tensor held
across the step. I wrap it in a `torch.autograd.Function` with the fixed signature
`fused_mlp_forward(x, w_fc, w_proj)`, accumulating the contraction in fp32 inside the kernel and
following the autocast dtype outside (the full module is in the answer).

Now the falsifiable expectations against the GELU rung's numbers, on both axes, because this rung bets
on both. On quality I expect a real drop below 2.2868: squared ReLU is the self-gating multiplicative
interaction the GLU variants credit for beating GELU, captured in this exact two-matrix slot, so
`val_loss`, `wikitext2_ppl` and `lambada_ppl` should all come in under the GELU floor — if they do not,
the "multiplicative interaction buys quality" story is wrong *here* and I should reconsider the
activation, not the kernel. On throughput the bet is riskier and I want to be honest about it. The
fusion *deletes* the activation's `2·MN` round-trip, which should help; but it does so by replacing
cuBLAS's exquisitely-tuned up-projection with a hand-rolled tiled matmul at fixed `BLOCK_M=BLOCK_N=64`,
`BLOCK_K=32`, and under `torch.compile` the GELU rung's matmuls were already heavily optimized. So the
real question this rung answers is whether the round-trip I save is worth more than the cuBLAS-vs-naive
matmul gap I take on. If `elapsed` comes in *at or below* 20035, the fusion paid; if it comes in
*above* 20035 even though quality improved, that is the diagnosis that the cleanest win keeps the
matmuls in cuBLAS and changes only the activation — i.e. the next rung should drop the Triton matmul
entirely and run squared ReLU in plain torch, betting that almost all the quality gain here comes from
the activation and almost none from the fusion.
