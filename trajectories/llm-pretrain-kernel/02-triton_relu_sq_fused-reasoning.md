The GELU floor landed where I expected on one axis and told me something sharp on the other. On
quality it is unremarkable by construction: `val_loss` 2.2868, `wikitext2_ppl` 43.94, `lambada_ppl`
68.2, and downstream `arc_easy` 54.67, `hellaswag` 33.61, `piqa` 63.44, `winogrande` 51.7 — exactly
the "unshaped GELU" level I predicted, the floor against which an *activation* change should move.
`winogrande` at 51.7 sits so close to chance for a 355M model that a point or two of movement on it is
noise, so I will lean on `val_loss` and the two perplexities as the quality verdict and treat
`arc_easy`/`hellaswag` as looser corroboration. On throughput it posted `elapsed` 20035, confirming
the other half of the prediction: the two matmuls plus the single HBM round-trip of the wide hidden
dominate, and tidying the activation's elementwise chain into one launch was respectable but not
transformative. So the GELU rung handed me two distinct openings. First, the activation is the quality
lever — GELU is asymptotically linear, a strongly-firing and a barely-firing unit pass through at
nearly the same slope, no super-linear shaping — and on LM perplexity that is famously only on par with
ReLU, so 2.2868 is what "no shaping beyond the knee" buys. Second, the activation is *still* a separate
HBM round-trip even after I fused its elementwise chain, because I deliberately left the matmuls in
torch — that ~1.07 GB of traffic per FFN per micro-batch the matmuls could in principle have absorbed.
This rung goes after *both* at once: change the activation to something with real super-linear shaping,
and execute it by folding it into the matmul that produces it.

Take the activation first, because the quality story has to lead. The defaults — ReLU, GELU, Swish —
are all, in their large-`x` behavior, the same animal: asymptotically linear. For big positive
pre-activation they pass the input through roughly proportionally, differing only near the origin in
how soft the knee is. That is a *choice*, and the GELU number says it is a choice that costs me —
2.2868 is the price of asymptotic linearity, and reshaping the knee is what GELU already did relative
to ReLU and it bought essentially nothing on perplexity. So a soft-knee tweak is deck-chairs; to move
the number I have to change the *asymptotics*, the behavior far from zero. There is an old result
staring at me from a different corner: Krotov and Hopfield, studying associative memory, used
rectified-polynomial energy terms `F(s) = sᵖ` on the positive branch, and noted the feed-forward
activation falling out of that energy is one degree lower — the `p = 2` energy case corresponds to
ReLU, the `p = 3` case to a rectified parabola — and they left an explicit open question hanging: past
the threshold, should the activation grow linearly, sub-linearly, or *faster than linearly*, and might
a higher rectified polynomial beat ReLU in ordinary networks? The whole pointwise literature answered
"linearly." The GELU floor I just measured is that answer's price tag. Let me take the other branch.

So the candidate is `act(z) = max(z, 0)²` — rectify, then square. Squared ReLU. Check it is not
obviously broken. Asymptotics: for large positive `z` it grows like `z²`, genuinely faster than linear
— a *different shape* from GELU, not a tweak of the same shape, which is exactly what I need. Below
zero it is flat zero, the same sparsity-inducing dead zone as ReLU. Numerics: in this FFN the
pre-activation is the output of a normalized-input linear layer under bf16 autocast, so it sits at
`O(1)`, and squaring an `O(1)` number is fine — even a moderate outlier of magnitude 4 becomes 16,
comfortably inside bf16's range. And crucially I am *not* tempted to go cubic or quartic: higher powers
amplify the tail viciously — in bf16 a pre-activation of 8 becomes 4096 at degree four — and the
gradient scales like `q·z^(q−1)`, so degree three gives quadratic gradient growth and degree four
cubic, both courting overflow and unstable updates. Degree two is the *minimal* super-linear rectified
polynomial: it leaves the linear regime by the smallest step, the forward growing only quadratically
and the gradient only linearly. That is the first reason to pick the square specifically and not just
"some higher power."

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
projections equal, `W = V`: it becomes `max(0, xW) ⊙ (xW)`, and `max(0, z)·z` is `z²` when `z > 0` and
`0` when `z ≤ 0` — exactly `max(0, z)² = relu(z)²`. So squared ReLU applied to `xW` *is* ReGLU with its
two gate matrices tied: the diagonal, weight-shared special case of the gated unit, the *same
multiplicative interaction* the gated-FFN literature credits for the gain, the unit gating against
itself. And it gets that interaction *for free in exactly the slot I am allowed to edit*: ReGLU needs
`W, V, W2` and the `2/3` shrink; squared ReLU keeps the original two matrices `w_fc, w_proj`, no extra
`V`, no narrower bottleneck. So the constraint that blocks GLU here is *precisely* what makes squared
ReLU the right move — the real reason the square is the sweet spot and not just "Krotov says higher is
sharper," the exact power at which `relu(z)·z` collapses into a self-gate. And "multiplicative" is not
a slogan: expanding `z² = (Σⱼ wⱼ xⱼ)² = Σⱼ,ₖ wⱼ wₖ xⱼ xₖ` shows a single unit computing an explicit sum
of all pairwise products of the input coordinates — a genuine bilinear form — from the one projection
the up-matmul already ran, which no univariate reshaping of GELU could ever produce.

I need the gradient before I can train it, and I want it clean. `f(z) = relu(z)² = (max(z,0))²`. For
`z > 0`, `f = z²`, `f'(z) = 2z = 2·relu(z)`; for `z < 0`, `f = 0 = 2·relu(z)`; at `z = 0` both pieces
meet at `0`. So the derivative is `f'(z) = 2·max(z, 0) = 2·relu(z)` everywhere, no case split in code —
and it is *continuous*, ramping smoothly through zero, unlike ReLU's derivative which jumps 0→1 at the
origin. Squared ReLU is `C¹`, a smoother optimization surface than ReLU while keeping the hard zero
below threshold. The gradient shape reinforces the forward shape: `2·relu(z)` is unbounded above, so a
hard-firing unit sends back a proportionally *larger* gradient — the optimizer is told which units
matter — whereas GELU's derivative saturates near 1 and treats a strong and a mild unit almost alike.
The activation derivative needs only the pre-activation; the down-projection weight gradient
`∂L/∂W2 = gᵀ @ post` needs the activated tensor `post`.

Now the throughput lever, and this is where the GELU rung's leftover round-trip gets paid off and where
the activation choice and the execution turn out entangled. The GELU floor left the activation as a
standalone HBM pass over the wide `(M, N)` hidden, `N = 4·n_embd`: the matmul wrote `pre` out, the
activation read `MN` and wrote `MN`, the second matmul read it again — ~1.07 GB per FFN per micro-batch,
~0.5 ms of pure bandwidth time on a ~2 TB/s bus, spent moving `pre` in and out of memory for no reason
other than that cuBLAS had nowhere to put it. Because the matmul already had every `pre` element
sitting in an fp32 register accumulator the instant it finished accumulating that output tile — and
instead of using it, cuBLAS wrote it to HBM. If I run the activation *on the accumulator, in registers,
before the store*, that `2·MN` of round-trip traffic simply vanishes. By the roofline picture it is
close to free, and I can size "close to free": the epilogue does about two flops per output element — a
`max` against zero and a multiply — while the matmul that produced that element did `2·K = 2048` flops
accumulating it over the `K = 1024` contraction, so the fused activation adds on the order of `2/2048`,
~0.1%, to the tile's arithmetic, running on data already live in registers with the tensor cores
otherwise idle between accumulation and store. I spend a tenth of a percent of extra compute to erase
the entire `2·MN` memory round-trip. The reason the GELU rung did not do this is that cuBLAS gives a
*fixed* epilogue (down-cast and store) with no hook to staple an activation onto the accumulator — it
is structural, not a tuning knob — so to fuse I have to write the up-projection matmul myself at the
tile level, where the epilogue is mine.

And here the entanglement pays: squared ReLU is the *easiest possible* activation to fuse. Softmax or
layernorm need a row reduction, so fusing them into a matmul epilogue would need the whole row resident,
crossing tile boundaries. `relu(z)²` is strictly local — one element at a time, no reduction, no
neighbor — so it folds into the per-tile epilogue with zero cross-tile communication: `relu(acc)²` on
the `BLOCK_M × BLOCK_N` accumulator is just an elementwise op on data I already hold. A gated unit like
ReGLU would also need `xV` computed and multiplied, more to fuse and more traffic; the self-gate needs
only the one accumulator. The quality lever and the throughput lever chose the same activation.

So the forward kernel tiles the up-projection: each program instance owns one `BLOCK_M × BLOCK_N`
output tile, loops over the contraction `K` in chunks of `BLOCK_K`, loading sub-tiles of `x` and
`w_fc.t()`, accumulating `acc += dot(a, b)` into an fp32 register accumulator, and after the K-loop the
epilogue computes `relu(acc)²`, down-casts to bf16, stores `post`. The fp32 accumulator is not
optional: the contraction sums `K = 1024` bf16 products into one output element, and bf16's ~8 mantissa
bits would let each late addition round away against a running sum that has grown large — a systematic
loss of the low-order bits — so keeping `acc` in fp32 and down-casting only once at the end is the
standard discipline, and it also hands me `pre` in full precision for the epilogue's square. With
`BLOCK_M = BLOCK_N = 64`, `BLOCK_K = 32` and `K = 1024` the K-loop runs 32 iterations per tile and the
grid is `cdiv(M, 64) × cdiv(N, 64) = 1024 × 64` tiles for one micro-batch — plenty to fill the SMs. I
keep the `offs_m < M` / `offs_n < N` masks and a masked K-tail so the kernel stays correct on a short
final micro-batch or a different width, even though on this exact shape everything divides evenly and no
mask fires. But I should be honest that these are *fixed* tile shapes, not autotuned. Each program
holds a `64×64` fp32 accumulator with one accumulator and synchronous loads, so the `tl.dot` on step
`k` cannot begin until that step's operands arrive and the load of step `k+1` cannot be issued while
those registers are live — the tensor cores sit idle across the memory latency of each of the 32
K-iterations, a stall I have no way to hide inside this loop. That idle time is exactly what a vendor
GEMM spends its complexity erasing: cuBLAS runs multi-stage asynchronous copies that prefetch the next
operands into shared memory while the current MMA is still accumulating, and chooses `BLOCK_K` and the
pipeline depth per shape. My kernel has none of that. So the honest pre-registration is that the naive
matmul could run at some unknown fraction of cuBLAS's rate here, set by how much of the K-loop is spent
stalling on loads, which I cannot bound from the armchair — the `elapsed` number is the only instrument
that reads it.

The second matmul `out = post @ w_proj.t()` has no activation after it (residual and dropout are
outside the block), so there is nothing to fuse into its epilogue — I leave it as a plain torch matmul,
the same decision the GELU rung made for *both* matmuls, now made for just the down one.

That framing exposes the bet as an asymmetric one, and I want it stated plainly before I commit. The
upside of the fusion is *bounded*: at most I delete the activation's round-trip, ~0.5 ms per FFN per
micro-batch, and no fusion can save more than that because it is the whole memory cost I am targeting.
The downside is *not* bounded the same way: I am replacing cuBLAS's up-projection with a hand-rolled
tiled matmul, and the gap between a naive fixed-tile GEMM and an autotuned vendor kernel on these FFN
shapes could be a small fraction or a multiple — open-ended, set by how far my scheduling falls short.
So I am wagering a bounded, known saving against an unbounded, unknown cost. That is a bet worth
*making* — because if the naive matmul is close to cuBLAS, I win the round-trip for nearly free — but
not one I should be confident about, and I want the number to settle it rather than my hope.

For backward I let the algebra dictate what to stash. Forward, with `x : (M, K)`, `w_fc : (N, K)` so
`N = 4·n_embd`, `w_proj : (d, N)`: `pre = x @ w_fc.t()`, `post = relu(pre)²`, `out = post @ w_proj.t()`.
Given `g = ∂L/∂out` of shape `(M, d)`: through the second matmul, `d_post = g @ w_proj` and
`∂L/∂w_proj = gᵀ @ post`; through the activation, `d_pre = 2·relu(pre) ⊙ d_post`; through the first
matmul, `∂L/∂x = d_pre @ w_fc` and `∂L/∂w_fc = d_preᵀ @ x`. Shapes confirm: `gᵀ (d, M) @ post (M, N) →
(d, N)` matches `w_proj`, and `d_preᵀ (N, M) @ x (M, K) → (N, K)` matches `w_fc`. The only
nonlinear-specific value the backward needs is `relu(pre)`, so I save just `pre` and recompute
`relu(pre)` and `relu(pre)²` from it in the backward — trading a cheap elementwise recompute for one
fewer wide tensor (`post`, `M·N` bf16, ~537 MB per FFN per micro-batch) held across the step. This is
the recompute trade I flagged at the GELU floor, now taken, because the activation here really is cheap
enough that holding `post` costs more than remaking it. I wrap it in a `torch.autograd.Function` with
the fixed signature `fused_mlp_forward(x, w_fc, w_proj)`, accumulating in fp32 inside the kernel and
following the autocast dtype outside (the full module is in the answer). One thing this backward buys
me for reading throughput: all four gradient matmuls stay plain torch/cuBLAS, and the forward
down-matmul too, so the *only* execution change from the floor is the single forward up-projection
kernel — if `elapsed` moves it is a clean single-cause measurement, exactly the isolation I need to
diagnose the fusion in the next rung rather than guess.

Now the falsifiable expectations against the GELU rung's numbers, because this rung bets on both axes.
On quality I expect a real drop below 2.2868: squared ReLU is the self-gating multiplicative
interaction the GLU variants credit for beating GELU, captured in this exact two-matrix slot, so
`val_loss`, `wikitext2_ppl` and `lambada_ppl` should all come in under the floor, and I would hope
`arc_easy`/`hellaswag` tick up in step — if none of the quality metrics move down, the "multiplicative
interaction buys quality" story is wrong *here* and I should reconsider the activation, not the kernel.
On throughput the bet is riskier, along the asymmetry I laid out: the fusion deletes the ~0.5 ms
round-trip but replaces cuBLAS's up-projection with a fixed-tile hand-rolled matmul at
`BLOCK_M=BLOCK_N=64`, `BLOCK_K=32`, and under `torch.compile` the GELU rung's matmuls were already
heavily optimized. If `elapsed` comes in *at or below* 20035, the fusion paid; if it comes in *above*
20035 even though quality improved, that is the diagnosis that the fusion was the detour and the
activation the whole prize — that almost all the quality gain here comes from the self-gating square and
almost none from folding it into the matmul, so the cleanest win would keep the matmuls in cuBLAS and
let only the activation change.
