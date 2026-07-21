I am standing at the default fill and I have to pick the first thing to change, knowing nothing yet about
how any of these moves actually land — there are no measured numbers in front of me. So I reason from the
structure of the scaffold and from what the substrate is paying for. The default `Block` runs two sequential
sublayers, each fronted by its own full `LayerNorm`, and I want to count that cost. A `LayerNorm` over a
token's feature vector is *two* reductions along the feature axis: one pass for the mean, a second for the
variance around it. Two norms per block is four reductions per block per token, 96 feature-axis reductions on
every token across the 24-layer stack, every one of the 12,030 iterations, on ~7B tokens. On top of that
there is a *serial* dependency: `x ← x + Attn(LN(x))` must finish and write back into the residual before
`x ← x + MLP(LN(x))` can read its input, because the MLP's `ln_2` reads the post-attention stream. So the
default block pays twice — duplicated normalization and a strict attention-then-MLP ordering — and neither
cost is obviously buying final-loss quality. Those are the two levers the edit surface exposes: the
normalization *rule* (`LayerNorm`) and the block *wiring* (`Block`). Everything else is frozen.

The one thing I set aside immediately is `CONFIG_OVERRIDES`: with no run yet, moving the learning rate or
schedule would change the measuring stick before I have measured anything, and a bad run afterward could not
be pinned on architecture versus optimizer. That leaves the three architectural fills — norm rule, wiring, or
both. I choose the bundle deliberately: the first rung plants the extreme of the ladder, the most aggressively
simplified block the surface allows, so everything afterward is measured against a known floor. One number as
a verdict on two changes is the messiest kind of experiment, and I pay for it by pinning the expected *sign*
of each half in advance: if the norm-rule half is quality-neutral and the wiring half is the part that costs,
the one number reads mostly as a verdict on the wiring, and this floor is what any less aggressive wiring is
later priced against. So the first rung is RMSNorm plus a parallel block, and I owe each half a defense.

Take the normalization rule first; it is the cleaner half and every rung after this inherits it. The default
`LayerNorm`, within one token's `a ∈ R^1024`, subtracts the mean `μ = (1/n)Σaᵢ`, divides by the standard
deviation `σ = √((1/n)Σ(aᵢ−μ)²)`, and applies a learned gain `γ` (and a bias `β`). Those buy two invariances:
subtracting the mean buys *re-centering* invariance — shift every coordinate by a constant, `μ` absorbs it,
`a−μ` is unchanged; dividing by `σ` buys *re-scaling* invariance — scale `a` by `α`, both `μ` and `σ` scale,
the ratio is untouched. The question that decides whether I can drop the mean is which invariance keeps
training well-conditioned, and mechanically it is the re-scaling one: stabilizing a 24-block stack is about
controlling the *spread* of activations and gradients across depth, and subtracting the mean recenters the
cloud but leaves the spread alone, `var(a−μ) = var(a)`. So I bet the mean-subtraction is dispensable and
replace `σ` — which is *defined* through `μ` — with a spread measure that references no mean, the root mean
square `RMS(a) = √((1/n)Σaᵢ²)`, giving `āᵢ = aᵢ/RMS(a) · γ`.

The two denominators are cleanly related: `σ² = (1/n)Σ(aᵢ−μ)² = RMS² − μ²`, so `RMS² = σ² + μ²` and
`RMS ≥ σ` always, with equality exactly when `μ = 0`. RMSNorm divides by something larger than `σ` by the
factor `√(1 + μ²/σ²)`, governed entirely by how a token's mean compares to its spread — `4.4%` larger at
`μ/σ = 0.3`, reaching `√2 ≈ 41%` only when mean and spread are comparable. So the operation I drop matters in
proportion to how off-center the activations run, and a pre-norm residual stream — whose whole job is to keep
activations well-behaved for the next block — has no structural reason to carry large per-token means; my
working estimate is that `RMS` and `σ` sit within a percent or two in practice, bounded by that identity even
though I cannot confirm it without instrumenting activations. When `μ = 0` the two layers coincide exactly, so
this is the same layer with recentering switched off, not a different mechanism; re-scaling invariance survives
cleanly because `RMS(αa) = α·RMS(a)`.

The bias goes with the mean, and the config makes the point: it sets `bias=False`, so the default fill has no
bias anyway — the right call for pre-norm, since a per-channel shift on a norm layer only exists to restore a
location after recentering, and with no recentering there is nothing to restore. So my RMSNorm keeps only the
gain `γ`, initialized to ones, and ignores the `bias` argument it is contractually handed. The capacity is
unchanged by the swap: the default `LayerNorm` here holds 49,152 gains and zero biases, RMSNorm holds the same
49,152 gains and zero biases. So swapping removes an *operation*, not a parameter — whatever quality difference
the number eventually shows between the two rules is attributable to the mean-subtraction and nothing else, a
rare clean attribution.

The backward pass is the real test, because the point of a norm is not its forward invariances but whether it
keeps the *gradient* conditioned through depth. `RMS` is quadratic in `a` and `a = Wx`, so the weight enters
both numerator and denominator. The Jacobian of the normalized vector with respect to `a` is
`R = (1/RMS(a))·(I − aaᵀ/(n·RMS(a)²))` — `1/RMS` times identity minus a rank-one projection of the radial
direction; it annihilates `a` itself (`R·a = 0`, since `aᵀa = n·RMS²`), as it must, because `a/RMS` is
scale-invariant. Chaining to the weight, `∂L/∂W = (R(γ ⊙ u))xᵀ` for upstream `u`, and scaling the input or the
weights by `δ` sends `R → R/δ`: the gradient is *invariant* to input scaling (the `δ` from `x` cancels the
`1/δ` from `R`) and *inversely proportional* to weight scaling (only `R` moves). A layer whose weights grow
large automatically receives smaller gradients — an implicit per-layer learning-rate adaptation that damps
further growth with no schedule and no extra parameters. None of that self-regulation came from the
mean-subtraction; it all lives in the re-scaling structure RMSNorm keeps. So RMSNorm is the load-bearing part
of `LayerNorm`, at one reduction instead of two, and it should cost nothing on quality.

One implementation detail I honor: the sum of squares accumulates in fp32 (`input.float()` before
`pow(2).mean`), because summing 1024 squared bf16 terms loses the tail; and `eps = 1e-5` sits inside the root,
`1/√(mean(a²)+eps)` — for unit-scale activations a ~1e-5 relative floor, negligible for the forward but there
to keep the `rsqrt` from exploding if a token's activations ever collapse toward zero. This is the rule every
later rung inherits, and I want its number on the board first.

Now the structural half, the larger and riskier one. In the default block the residual is updated twice in
series — the MLP reads the post-attention residual, 48 sequential sublayer writes across the model. The
parallel block reads *one* normalized copy of the pre-block residual and lets both sublayers operate on it:
`h = LN(x); x ← x + Attn(h) + MLP(h)` — one shared norm per block instead of two, taking the stack from 48
feature-axis reductions per token down to 24. But I should be honest about whether "fewer reductions" is really
a *speed* win, because the arithmetic does not support the intuition. A norm reduction touches `d = 1024`
values per token; a single MLP projection is a `1024×4096` matmul, ~4.19M multiply-adds. One norm is ~0.024%
of *one* of the block's several large matmuls — removing it saves essentially no FLOPs. A norm costs wall-clock
only through memory bandwidth: a low-intensity kernel that streams the full `N×d` activation tensor twice
(once to reduce, once to apply) at a fraction of peak, plus launch overhead. And the other advertised parallel
win — the shortened critical path from running attention and MLP concurrently — is only realized if the two
branches actually fuse, which at large scale comes from fusing the attention and MLP *input projections* into
one matmul; that fusion lives inside the frozen `CausalSelfAttention`/`MLP`, outside my edit surface. I can
share the norm and drop the write-ordering, but the two matmul-heavy branches still launch back to back. So my
honest prediction is the *fastest* rung, but only by the memory traffic of the one norm it drops — a few
percent, not a halving. If a later non-parallel block keeping this RMSNorm comes in only slightly slower, that
confirms the speed win was just the norm; if the parallel block is *dramatically* faster, I was wrong about the
projections not overlapping.

What parallelizing costs *representationally* is why I expect this to be the weakest rung on quality, and the
mechanism is specific. Attention *mixes across positions* — for each token, a context-weighted sum of value
vectors from other tokens — while the MLP *mixes across channels* at a fixed position, a per-token nonlinear
transform. In the sequential block the MLP at block ℓ transforms features that already contain the context
attention just gathered at block ℓ: a single layer can express "gather this context, then nonlinearly
transform it." In the parallel block that composition is severed — the MLP at ℓ transforms the *pre-attention*
`h`, and attention's fresh context is folded in only at ℓ+1, one layer later. The model can still express it,
but it costs a *layer of depth* to do so, and with only 24 layers that is a non-trivial fraction of the budget.
This is the known small-scale tax of the parallel block: negligible at very large depth and width, where the
one-layer lag is a rounding error against total capacity, but a 355M model is firmly in the regime where it
should bite. There is a second, subtler risk from *combining* the changes: with one shared RMSNorm and a
summed residual `x + Attn(h) + MLP(h)`, two branch outputs add in with no intervening normalization, and
RMSNorm does not recenter, so any mean drift in `h` is no longer cleaned up before feeding two branches. In a
pre-norm stream whose variance already grows with depth, that could run a little hotter — how much, this rung
exists to find out.

The one way this could go badly wrong rather than merely suboptimally is the initialization contract, tuned
for the sequential block. The fixed init scales every `c_proj` by `1/√(2·n_layer) = 1/√48 ≈ 0.1443` so the
variance each block adds to the residual stays bounded; the `2` is there because the sequential block writes
to the residual twice per block, `48` writes total, keeping the added variance `O(1)` and the residual RMS
near `√2×` its input rather than `√48×`. The parallel block also sums two writes per block, `48` total — so the
`1/√48` still matches the write count, each write is still scaled by `1/48` in variance, and the total added
variance is still `O(1)`. The wiring change does not silently break the variance calibration the substrate
depends on; the risk here is the representational one — lost cross-talk and a possibly hotter mean drift — not
an init blow-up.

So the first rung is settled, a clean fill of the two-region surface: replace `LayerNorm` with an RMSNorm that
keeps only the gain and ignores the `bias` argument, and replace `Block` with a single shared-norm parallel
block that sums attention and MLP into the residual in one step. I touch nothing in `CONFIG_OVERRIDES`. The
distilled module and the literal scaffold code are in the answer.

What I can predict a priori, against the only anchor I have — the default fill this replaces. RMSNorm alone
should be roughly quality-neutral: it drops an operation I argued is not load-bearing and changes no
parameters. The parallel wiring is where I expect the cost — at 355M the lost intra-block cross-talk, plus the
summed un-normalized branches, should read as a *slightly higher* `val_loss` than a non-parallel RMSNorm block
would give — paid back only by a *modest* speed win, the memory traffic of one dropped norm, since the
critical-path fusion is outside my reach. So: trains stably to completion, cheapest per iteration by a few
percent, competitive but not best. WikiText-2 and LAMBADA perplexities should track `val_loss` in direction;
LAMBADA leans hardest on attention's long-range context, so the one-layer lag could bite a touch more there.
ARC-Easy and HellaSwag are the coarsest witnesses at this scale and I would not read small moves either way. I
run this first to price the parallel tax — the gap between this floor and a later block that keeps the RMSNorm
but does not parallelize is exactly what going parallel costs.
