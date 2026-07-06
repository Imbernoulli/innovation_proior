The GELU floor landed where I expected on one axis and told me something sharp on the other. On
quality it is unremarkable by construction: `val_loss` 2.2868, `wikitext2_ppl` 43.94, `lambada_ppl`
68.2, and downstream `arc_easy` 54.67, `hellaswag` 33.61, `piqa` 63.44, `winogrande` 51.7 — exactly
the "unshaped GELU" level I predicted, the floor against which an *activation* change should move.
Those downstream numbers are worth writing down as the reference they are: `arc_easy` in the mid-54s,
`hellaswag` in the low-33s, `winogrande` near chance for a 355M model — I will read any later shift
against these, and I note now that `winogrande` sits so close to 50 that a point or two of movement on
it is noise, not signal, so I will lean on `val_loss` and the two perplexities as the quality verdict
and treat `arc_easy`/`hellaswag` as a looser corroboration. On throughput it posted `elapsed` 20035 leaving the matmuls to cuBLAS, which confirms
the other half of the prediction: the two matmuls plus the single HBM round-trip of the wide hidden
dominate, and tidying the activation's elementwise chain into one launch was respectable but not
transformative. So the GELU rung handed me two distinct openings. First, the activation is the quality
lever — GELU is asymptotically linear, a strongly-firing and a barely-firing unit pass through at
nearly the same slope, no super-linear shaping — and on LM perplexity that is famously only on par with
ReLU, so 2.2868 is what "no shaping beyond the knee" buys. Second, the activation is *still* a separate
HBM round-trip even after I fused its elementwise chain, because I deliberately left the matmuls in
torch — that ~537 MB of hidden read and written back per FFN per micro-batch, ~1.07 GB of traffic the
matmuls could in principle have absorbed. This rung goes after *both* at once: change the activation to
something with real super-linear shaping, and execute it by folding it into the matmul that produces
it, so the round-trip the GELU rung left on the table disappears too.

Take the activation first, because the quality story has to lead. The defaults — ReLU, GELU, Swish —
are all, in their large-`x` behavior, the same animal: asymptotically linear. For big positive
pre-activation they pass the input through roughly proportionally, differing only near the origin in
how soft the knee is. That is a *choice*, and the GELU number says it is a choice that costs me —
2.2868 is the price of asymptotic linearity, and reshaping the knee is what GELU already did relative
to ReLU and it bought essentially nothing on perplexity. So a soft-knee tweak is deck-chairs; if I want
to move the number I have to change the *asymptotics*, the behavior of the function far from zero. There
is an old result staring at me from a different corner: Krotov and Hopfield, studying associative
memory, used rectified-polynomial energy terms `F(s) = sᵖ` on the positive branch, and noted that the
feed-forward activation falling out of that energy is one degree lower — the `p = 2` energy case
corresponds to ReLU, the `p = 3` case to a rectified parabola — and they left an explicit open question
hanging: past the threshold, should the activation grow linearly, sub-linearly, or *faster than
linearly*, and might a higher rectified polynomial beat ReLU in ordinary networks? The whole pointwise
literature answered "linearly." The GELU floor I just measured is that answer's price tag. Let me take
the other branch.

So the candidate is `act(z) = max(z, 0)²` — rectify, then square. Squared ReLU. Before getting
excited, check it is not obviously broken. Asymptotics: for large positive `z` it grows like `z²`,
genuinely faster than linear — a *different shape* from GELU, not a tweak of the same shape, which is
exactly what I need. Below zero it is flat zero, the same sparsity-inducing dead zone as ReLU.
Numerics: will `z²` blow up? In this FFN the pre-activation is the output of a normalized-input linear
layer under bf16 autocast, so it sits at `O(1)`; squaring an `O(1)` number is fine, and even a moderate
outlier of magnitude 4 becomes 16, still comfortably inside bf16's range. And crucially I am *not*
tempted to go cubic or quartic, because higher powers amplify the tail viciously — in bf16 a
pre-activation of 8 becomes 4096 at degree four, and the gradient scales like `q·z^(q−1)`, so at degree
three the gradient is `3z²` (quadratic growth) and at degree four `4z³` (cubic growth), both courting
overflow and unstable updates. Degree two is the *minimal* super-linear rectified polynomial: it leaves
the linear regime by the smallest step, keeping the numerics tame — the forward grows only quadratically
and the gradient only linearly — while getting the sharper nonlinearity. That is the first reason to
pick the square specifically and not just "some higher power."

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

I can make the "multiplicative interaction" claim fully concrete by expanding the square. On the active
branch the pre-activation is `z = xW = Σⱼ wⱼ xⱼ`, and squaring it gives
`z² = (Σⱼ wⱼ xⱼ)² = Σⱼ,ₖ wⱼ wₖ xⱼ xₖ` — an explicit sum over *all pairwise products* `xⱼ xₖ` of the
input coordinates, weighted by `wⱼ wₖ`. So a single squared-ReLU unit computes a genuine bilinear form
in the input, a second-order feature interaction, from the one projection the up-matmul already ran. A
plain `GELU(xW)` cannot: it is a fixed univariate function of the *scalar* `z`, so it can bend that one
number but never form a product of two different input directions. That expansion is the mechanism the
gated FFNs are named for, appearing here with the diagonal weight-sharing `wⱼ wₖ` instead of ReGLU's
independent `wⱼ vₖ` — fewer free products, but still products, and still for free in the two-matrix
slot. It is also why "just make the knee softer" was always going to fail on perplexity: no reshaping
of a univariate curve can manufacture a cross-term.

Let me make concrete what this actually does to the function, because "super-linear shaping" is a phrase
and I want the shape in numbers. Put squared ReLU next to GELU at a few pre-activations. At `z = 0.5`:
GELU gives `0.346`, squared ReLU gives `0.25` — squared ReLU is *smaller*, it suppresses a weakly-firing
unit relative to GELU. At `z = 2`: GELU gives `1.95`, squared ReLU gives `4` — now it is more than
double, it amplifies a strongly-firing unit. The two curves cross somewhere in between; solving
`z² = GELU(z)` on the positive branch, `z = 0.7` gives `0.49` versus `0.53` (squared ReLU still below)
and `z = 0.8` gives `0.64` versus `0.63` (squared ReLU just above), so they cross near `z ≈ 0.8`. That
crossover is the whole shape in one number: below it squared ReLU pushes weak units *down* toward the
dead zone, above it drives strong units *up* faster than linear, and the gap widens quadratically as
`z` grows. This is not a reshaped knee — the knee governs a small neighborhood of the origin — it is a
reshaped *tail*, a sharpening that says "confident units count more, hesitant ones count less," which
is the multiplicative self-gate made visible. That contrast is exactly why I expect it to move a number
GELU could not.

I need the gradient before I can train it, and I want it clean. `f(z) = relu(z)² = (max(z,0))²`. For
`z > 0`, `f = z²`, `f'(z) = 2z = 2·relu(z)` since `relu(z) = z` there. For `z < 0`, `f = 0`,
`f'(z) = 0 = 2·relu(z)` since `relu(z) = 0` there. At `z = 0` both pieces meet at `0`, and
`2·relu(0) = 0`, no jump. So the derivative is just `f'(z) = 2·max(z, 0) = 2·relu(z)` everywhere, no
case split in code — and notice it is *continuous*, ramping smoothly through zero, unlike ReLU's
derivative which jumps 0→1 at the origin. Squared ReLU is `C¹`, a smoother optimization surface than
ReLU while keeping the hard zero below threshold. A quiet bonus. And the gradient shape reinforces the
forward shape: `2·relu(z)` is unbounded above, so a hard-firing unit sends back a proportionally
*larger* gradient — the optimizer is told which units matter — whereas GELU's derivative saturates near
1 and treats a strong and a mild unit almost alike. The activation derivative needs only the
pre-activation; the down-projection weight gradient `∂L/∂W2 = gᵀ @ post` needs the activated tensor
`post`.

Now the throughput lever, and this is where the GELU rung's leftover round-trip gets paid off and where
the activation choice and the execution turn out entangled. The GELU floor left the activation as a
standalone HBM pass over the wide `(M, N)` hidden, `N = 4·n_embd`: the matmul wrote `pre` out, the
activation read `MN`, wrote `MN`, the second matmul read it again. Concretely that read-plus-write is
~1.07 GB per FFN per micro-batch, ~0.5 ms of pure bandwidth time on a ~2 TB/s bus, spent moving `pre`
in and out of memory for no reason other than that cuBLAS had nowhere to put it. Because the matmul
already had every `pre` element sitting in an fp32 register accumulator the instant it finished
accumulating that output tile — and instead of using it, cuBLAS wrote it to HBM. If I run the activation
*on the accumulator, in registers, before the store*, that `2·MN` of round-trip traffic — the dominant
cost of the bandwidth-bound activation step — simply vanishes. By the roofline picture it is close to
free, and I can put a number on "close to free": the epilogue does about two flops per output element —
a `max` against zero and a multiply — while the matmul that produced that element did `2·K = 2048` flops
accumulating it over the `K = 1024` contraction. So the fused activation adds on the order of `2/2048`,
~0.1%, to the tile's arithmetic, and that sliver runs on data already live in registers with the tensor
cores otherwise idle between accumulation and store. I spend a tenth of a percent of extra compute to
erase the entire `2·MN` memory round-trip. The
reason the GELU rung did not do this is that cuBLAS gives a *fixed* epilogue (down-cast and store) with
no hook to staple an activation onto the accumulator — it is structural, not a tuning knob — so to fuse
I have to write the up-projection matmul myself at the tile level, where the epilogue is mine.

And here the entanglement pays: squared ReLU is the *easiest possible* activation to fuse. Softmax or
layernorm need a row reduction, so fusing them into a matmul epilogue would need the whole row resident,
crossing tile boundaries. `relu(z)²` is strictly local — one element at a time, no reduction, no
neighbor — so it folds into the per-tile epilogue with zero cross-tile communication: `relu(acc)²` on
the `BLOCK_M × BLOCK_N` accumulator is just an elementwise op on data I already hold. A gated unit like
ReGLU would also need `xV` computed and multiplied, more to fuse and more traffic; the self-gate needs
only the one accumulator. The quality lever and the throughput lever chose the same activation.

So the forward kernel tiles the up-projection: each program instance owns one `BLOCK_M × BLOCK_N`
output tile, loops over the contraction `K` in chunks of `BLOCK_K`, loading sub-tiles of `x` and
`w_fc.t()`, accumulating `acc += dot(a, b)` into an fp32 register accumulator, and after the K-loop the epilogue
computes `relu(acc)²`, down-casts to bf16, stores `post`. The fp32 accumulator is not optional: the
contraction sums `K = 1024` bf16 products into one output element, and bf16 carries only ~8 mantissa
bits, so accumulating a thousand terms in bf16 would let each late addition round away against a running
sum that has grown large — a systematic loss of the low-order bits precisely where the sum is most
sensitive. Keeping `acc` in fp32 (23 mantissa bits) and down-casting only once at the end is the
standard discipline, and it also happens to be what hands me `pre` in full precision for the epilogue's
square. With `BLOCK_M = BLOCK_N = 64`, `BLOCK_K = 32` and `K = n_embd = 1024`, the K-loop runs
`1024/32 = 32` iterations per tile, and the grid is `cdiv(M, 64) × cdiv(N, 64) = 1024 × 64 = 65,536`
tiles for one micro-batch — plenty to fill the SMs. Let me trace the indexing once to be sure the
kernel is correct on this shape and degrades gracefully off it, since a matmul with a mis-masked tail is
a silent training bug. For this micro-batch `M = 65,536` is `1024 × 64` and `N = 4096` is `64 × 64` on
the nose, so `cdiv` divides evenly and every one of the 65,536 tiles is full — no partial tile, the
`offs_m < M` and `offs_n < N` masks never fire. The K-loop `for k in range(0, 1024, 32)` runs
`k = 0, 32, …, 992`, exactly 32 iterations, the last covering `offs_k = 992…1023`, all `< K`, so the
contraction sums all 1024 terms with no over-read. Off this exact shape the masks earn their keep: a
short final micro-batch with, say, `M = 65,500` gives `cdiv(65,500, 64) = 1024` row-tiles, the last
covering rows `65,472…65,535` of which only `65,500 − 65,472 = 28` are valid — the `offs_m < M` mask
loads zero for the other 36 rows so they never touch `acc` and are never stored. Correct for the shape I
run, correct for any shape I might. But I should be honest that these are *fixed* tile
shapes, not autotuned, and the kernel has none of the software pipelining, double-buffering, or
shape-specialized scheduling that a vendor GEMM brings; it is a correct tiled matmul, not a fast one.

Let me make "not a fast one" concrete, because the size of that gap is the whole risk of this rung and
I would rather predict it than be surprised by it. Each program holds a `64×64` fp32 accumulator — 4096
registers, 16 KB of register file for that tile alone — and every K-step loads a `64×32` sub-tile of `x`
and a `32×64` sub-tile of `w_fcᵀ`, ~4 KB each in bf16. With one accumulator and no double-buffering, the
`tl.dot` on step `k` cannot begin until that step's operands have arrived, and the load of step `k+1`
cannot be issued while those registers are still live — so the tensor cores sit idle across the memory
latency of each of the 32 K-iterations, a stall I have no way to hide inside this loop. That idle time is
exactly what a vendor GEMM spends its complexity erasing: cuBLAS runs multi-stage asynchronous copies
that prefetch the next operands into shared memory while the current MMA is still accumulating, so the
tensor cores never wait; it chooses `BLOCK_K` and the number of pipeline stages to match the
architecture's latency and blocks the operands through register and shared-memory tiers tuned per shape.
My kernel has none of that — one accumulator, synchronous loads, a fixed `64/64/32` tiling I picked by
eye rather than autotuned against these exact `(M, N, K)`. So the honest pre-registration is that the
naive matmul could run at some unknown fraction of cuBLAS's rate here, the fraction set by how much of
the K-loop is spent stalling on loads — which I cannot bound from the armchair. The `elapsed` number is
the only instrument that reads it.

The second matmul `out = post @ w_proj.t()` has no activation after it (residual and dropout are outside
the block), so there is nothing to fuse into its epilogue — I leave it as a plain torch matmul, the
same decision the GELU rung made for *both* matmuls, now made for just the down one.

That framing exposes the bet as an asymmetric one, and I want it stated plainly before I commit. The
upside of the fusion is *bounded*: at most I delete the activation's round-trip, ~0.5 ms per FFN per
micro-batch, and no fusion can save more than that because it is the whole memory cost I am targeting.
The downside is *not* bounded the same way: I am replacing cuBLAS's up-projection with a hand-rolled
tiled matmul, and the gap between a naive fixed-tile GEMM and an autotuned vendor kernel on these FFN
shapes could be a small fraction or it could be a multiple — it is set by how far my scheduling falls
short, which is open-ended. So I am wagering a bounded, known saving against an unbounded, unknown cost.
That is a bet worth *making* — because if the naive matmul is close to cuBLAS, I win the round-trip for
nearly free — but it is not a bet I should be confident about, and I want the number to settle it rather
than my hope.

For backward I keep it minimal and let the algebra dictate what to stash. Forward, with `x : (M, K)`,
`w_fc : (N, K)` so `N = 4·n_embd`, `w_proj : (d, N)`: `pre = x @ w_fc.t()`, `post = relu(pre)²`,
`out = post @ w_proj.t()`. Given `g = ∂L/∂out` of shape `(M, d)`: through the second matmul,
`d_post = g @ w_proj` shape `(M, N)`, and `∂L/∂w_proj = gᵀ @ post` shape `(d, N)`; through the
activation, `d_pre = 2·relu(pre) ⊙ d_post`; through the first matmul, `∂L/∂x = d_pre @ w_fc` and
`∂L/∂w_fc = d_preᵀ @ x`. Let me confirm the two weight-gradient shapes so I trust them:
`gᵀ (d, M) @ post (M, N) → (d, N)` matches `w_proj`, and `d_preᵀ (N, M) @ x (M, K) → (N, K)` matches
`w_fc`. Good. The only nonlinear-specific value the backward needs is `relu(pre)`, so I save just `pre`
and recompute `relu(pre)` and `relu(pre)²` from it in the backward — `relu(pre)²` for the `w_proj`
gradient and `2·relu(pre)` for the activation gradient. Saving only `pre` and recomputing, rather than
also stashing `post`, trades a cheap elementwise recompute for one fewer wide tensor held across the
step: `post` is `M·N` bf16, ~537 MB per FFN per micro-batch, and `relu` plus a multiply to regenerate
it in the backward is trivial next to the matmuls happening around it — so this is the recompute trade
I flagged at the GELU floor, now taken, because the activation here really is cheap enough that holding
`post` costs more than remaking it. I wrap it in a `torch.autograd.Function` with the fixed signature
`fused_mlp_forward(x, w_fc, w_proj)`, accumulating the contraction in fp32 inside the kernel and
following the autocast dtype outside (the full module is in the answer). One thing this backward buys
me for reading the throughput result: all four gradient matmuls (`d_post = g @ w_proj`,
`∂L/∂w_proj = gᵀ @ post`, `∂L/∂x = d_pre @ w_fc`, `∂L/∂w_fc = d_preᵀ @ x`) stay plain torch, so they go
to cuBLAS exactly as in the GELU rung, and the down-projection in the forward is torch too. The *only*
thing that changed from the floor's execution is the single forward up-projection, now my hand-rolled
kernel. So if `elapsed` moves, it is a clean single-cause measurement — the delta is attributable to
that one kernel and nothing else, which is exactly the isolation I need to diagnose the fusion in the
next rung rather than guess.

Now the falsifiable expectations against the GELU rung's numbers, on both axes, because this rung bets
on both. On quality I expect a real drop below 2.2868: squared ReLU is the self-gating multiplicative
interaction the GLU variants credit for beating GELU, captured in this exact two-matrix slot, so
`val_loss`, `wikitext2_ppl` and `lambada_ppl` should all come in under the GELU floor, and I would hope
the downstream reads (`arc_easy`, `hellaswag`) tick up in step — if none of the quality metrics move
down, the "multiplicative interaction buys quality" story is wrong *here* and I should reconsider the
activation, not the kernel. On throughput the bet is riskier and I want to be honest about it, exactly
along the asymmetry I laid out. The fusion *deletes* the activation's ~0.5 ms round-trip, which should
help; but it does so by replacing cuBLAS's exquisitely-tuned up-projection with a hand-rolled tiled
matmul at fixed `BLOCK_M=BLOCK_N=64`, `BLOCK_K=32`, and under `torch.compile` the GELU rung's matmuls
were already heavily optimized. So the real question this rung answers is whether the round-trip I save
is worth more than the cuBLAS-vs-naive matmul gap I take on. If `elapsed` comes in *at or below* 20035,
the fusion paid; if it comes in *above* 20035 even though quality improved, that is the diagnosis that
the fusion was the detour and the activation was the whole prize — that almost all the quality gain here
comes from the self-gating square and almost none from folding it into the matmul, so the cleanest win
would keep the matmuls in cuBLAS and let only the activation change.
