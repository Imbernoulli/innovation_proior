I am standing at the default fill and I have to pick the first thing to change, knowing nothing yet about
how any of these moves actually land — there are no measured numbers in front of me. So I reason from the
structure of the scaffold and from what the substrate is paying for. The default `Block` runs two sequential
sublayers, each fronted by its own full `LayerNorm`, and I want to count that cost concretely before I decide
what to attack. A `LayerNorm` over a token's feature vector is *two* reductions along the feature axis: one
pass to compute the mean, a second to compute the variance around that mean. Two norms per block therefore
cost four reductions per block per token, and across the 24-layer stack that is 96 feature-axis reductions
on every token, every one of the 12,030 iterations, on ~7B tokens. On top of the reductions there is a
*serial* dependency baked into the block: `x ← x + Attn(LN(x))` must finish and write back into the residual
before `x ← x + MLP(LN(x))` can even read its input, because the MLP's `ln_2` reads the post-attention
stream. So the default block is paying twice — once for duplicated normalization, once for a strict
attention-then-MLP ordering — and neither cost is obviously buying me final-loss quality. Those are the two
levers the edit surface actually exposes: the normalization *rule* (the `LayerNorm` class) and the block
*wiring* (the `Block` class). Everything else is frozen.

Let me lay out the moves those two levers allow before committing, because the leaderboard will hand me back
exactly one number and I need to know in advance what that number is a verdict on. I could change (a) only
the norm rule, keeping the sequential wiring; (b) only the wiring, keeping `LayerNorm`; (c) both at once; or
(d) reach into `CONFIG_OVERRIDES` and move the learning rate or schedule. Option (d) I set aside immediately:
I have no diagnostic yet, not a single run, and the learning rate, warmup, and cosine schedule were chosen
for *this* 355M model on *this* data — moving them now would be changing the measuring stick before I have
measured anything, and if a run then looked bad I could not tell whether the architecture or the optimizer
did it. That leaves the three architectural fills, and I am going to choose (c), the bundle, deliberately —
not because it is the cleanest experiment (it is the messiest, one number as a verdict on two changes) but
because the *purpose* of the first rung is to plant the extreme of the ladder: the most aggressively
simplified block the surface allows, so that everything I try afterward is measured against a known floor. I
accept the disentangling cost by pinning down, in advance, the expected *sign* of each half — if I can argue
that the norm-rule half is roughly quality-neutral and the wiring half is the part that costs, then the one
number reads mostly as a verdict on the wiring, and this floor is what any less aggressive wiring of the
two sublayers would later be priced against. So the first rung is RMSNorm plus a parallel block,
and I owe myself a defense of each half.

Take the normalization rule first, because it is the cleaner half and it is the one every rung after this
inherits. The default `LayerNorm`, within a single token's `a ∈ R^1024`, computes the mean `μ = (1/n)Σ aᵢ`,
subtracts it, divides by the standard deviation `σ = √((1/n)Σ(aᵢ−μ)²)`, and applies a learned per-channel
gain `γ` (and a bias `β`, which I will get to). That is two operations welded together, and they buy two
different invariances. Subtracting the mean buys *re-centering invariance*: shift every coordinate of `a` by
the same constant and the output is unchanged, because `μ` shifts by exactly that constant and the centered
vector `a − μ` does not move. Dividing by `σ` buys *re-scaling invariance*: multiply `a` by a positive `α`
and both `μ` and `σ` scale by `α`, so `(a−μ)/σ` is untouched. Two operations, two invariances, cleanly
separable. The question that decides whether I can drop the mean is: which invariance keeps training
well-conditioned? Mechanically the answer is the re-scaling one. Stabilizing a 24-block residual stack is
about controlling the *spread* of activations and gradients so they neither blow up nor vanish across depth;
subtracting the mean recenters the cloud but does nothing to its spread, because `var(a − μ) = var(a)`. It
tidies the location, it does not touch the scale. The operation that actually pins the magnitude the next
block and the backward pass see is the division by the scale. So I bet the mean-subtraction is dispensable
and replace `σ` — which is *defined* through `μ` — with a measure of spread around the origin that references
no mean at all, the root mean square `RMS(a) = √((1/n)Σ aᵢ²)`, giving `āᵢ = aᵢ/RMS(a) · γ`.

I want to know exactly what I am throwing away, so let me relate the two denominators rather than wave at
them. Expanding the variance, `σ² = (1/n)Σ(aᵢ−μ)² = (1/n)Σaᵢ² − μ² = RMS² − μ²`. That is a clean identity:
`RMS² = σ² + μ²`, so `RMS ≥ σ` always, with equality precisely when `μ = 0`. RMSNorm therefore divides by
something a hair larger than `σ`, larger by the factor `√(1 + μ²/σ²)` — and when the mean of a token's
features happens to be zero the two layers are *identical*. This is not a wild departure; it is the same
layer with the recentering switched off, differing only by however much per-token mean the activations carry.
Re-scaling invariance survives cleanly because `RMS(αa) = α·RMS(a)` is linear, exactly the property the
argument needs; only re-centering is discarded, the invariance I argued does not matter for spread control.

Let me put a number on "a hair larger" so I know the size of what I am discarding rather than trust the word.
The factor `√(1 + μ²/σ²)` is governed entirely by how a token's mean compares to its spread. Take a small
worked vector to feel it: `a = [1.2, −0.8, 0.3, −0.5]`. Its mean is `μ = 0.2/4 = 0.05`, its mean-square is
`(1.44 + 0.64 + 0.09 + 0.25)/4 = 0.605` so `RMS = 0.7778`, and its variance is `0.605 − 0.05² = 0.6025` so
`σ = 0.7762`. The two denominators differ by `RMS/σ = 1.0021` — two parts in a thousand — because this vector
is nearly centered, `μ²/σ² = 0.0041`. Push the mean off and the gap opens on a known curve: a token with
`μ/σ = 0.3` gives `√1.09 = 1.044`, `RMS` `4.4%` above `σ`, and only at `μ/σ = 1`, mean and spread comparable,
does it reach `√2 ≈ 41%`. So the operation I am dropping matters in proportion to how off-center the
activations run, and a pre-norm residual stream — whose whole job is to keep activations well-behaved for the
next block — has no structural reason to carry large per-token means. My working estimate is that `RMS` and `σ`
sit within a percent or two of each other in practice; whether that holds I cannot confirm without
instrumenting the activations, but the identity bounds it and the pre-norm setting makes small means the
expected case.

The bias goes with the mean, and here the config makes the point for me. The default `LayerNorm` class
carries a `β`, but the config sets `bias=False`, so the actual default fill has *no bias anyway* — and that
is the right call for a pre-norm transformer, because the only reason to carry a per-channel shift on a
normalization layer is to restore a location after recentering, and if I am not recentering there is nothing
to restore. So my RMSNorm keeps only the gain `γ`, initialized to ones, and ignores the `bias` argument it is
contractually handed. The parameter accounting is worth doing because it sharpens what this change *is*: the
default `LayerNorm` here holds one gain vector of length 1024 per norm, two norms per block, 24 blocks —
49,152 gain parameters and zero biases. RMSNorm holds exactly the same 49,152 gains and zero biases. The
counts are *identical*. So swapping `LayerNorm` for RMSNorm changes no capacity at all — it removes an
*operation* (the mean-subtraction), not a parameter. Whatever quality difference the number eventually shows
between these two rules is attributable to that single operation and nothing else, which is a rare clean
attribution and part of why I trust this half.

The backward pass is the real test, because the point of a norm is not its forward invariances but whether it
keeps the *gradient* conditioned through depth. Because `RMS` is quadratic in `a` and `a = Wx`, the weight
enters both numerator and denominator. Writing out the Jacobian of the normalized vector with respect to `a`
gives `R = (1/RMS(a))·(I − aaᵀ/(n·RMS(a)²))` — `1/RMS` times identity minus a rank-one outer product that
projects out the radial direction. I can check that this is the right shape by feeding it the one input it
must annihilate: perturbing `a` along `a` itself only rescales `a`, and `a/RMS` is scale-invariant, so `R·a`
must be zero. Compute it: `R·a = (1/RMS)(a − a·(aᵀa)/(n·RMS²))`, and since `aᵀa = n·RMS²` by definition of
`RMS`, the bracket is `a − a = 0`. It vanishes exactly. Good — the Jacobian kills the direction it must and
passes everything orthogonal through scaled by `1/RMS`. Chaining to the weight, `∂L/∂W = (R(γ ⊙ u))xᵀ` for
upstream `u`, and scaling the input or the weights by `δ` sends `R → R/δ`: the gradient comes out *invariant*
to input scaling (the `δ` from `x` cancels the `1/δ` from `R`) and *inversely proportional* to weight scaling
(only `R` moves). A layer whose weights have grown large automatically receives smaller gradients — an
implicit per-layer learning-rate adaptation that falls straight out of the quadratic form and damps further
growth with no schedule and no extra parameters.

To make that self-regulation concrete rather than a claimed proportionality: suppose during training one
layer's weights grow so that `a = Wx` doubles in scale, `δ = 2`. Then `RMS(a)` doubles with it, the Jacobian
`R = (1/RMS(a))·(I − aaᵀ/(n·RMS²))` halves, and `∂L/∂W = (R(γ ⊙ u))xᵀ` inherits that halving — the layer that
grew its weights `2×` receives gradients `0.5×` as large, a built-in brake with no line in the schedule. Apply
the same doubling to the *input* `x` instead and the gradient is untouched, because the `δ` carried in `xᵀ`
cancels the `1/δ` in `R`. That asymmetry — invariant to input scale, inverse to weight scale — is the whole of
the implicit per-layer learning-rate control, and it is a property of the quadratic denominator, so it is
exactly the part RMSNorm keeps and the mean-subtraction never contributed to.

None of that self-regulation came from the mean I am
discarding; all of it lives in the re-scaling structure I keep. So RMSNorm is strictly the load-bearing part
of `LayerNorm`, at one reduction instead of two, and — the part I care about for quality — it should cost me
nothing, because the piece removed was not controlling the spread. One implementation detail I will honor:
the sum of squares must be accumulated in fp32 (`input.float()` before `pow(2).mean`) because summing 1024
squared bf16 terms loses precision in the tail, and the `eps = 1e-5` sits inside the root, `1/√(mean(a²)+eps)`
— for unit-scale activations `mean(a²) ≈ 1`, so eps is a ~1e-5 relative floor, negligible for the forward but
there to keep the `rsqrt` from exploding if a token's activations ever collapse toward zero. This is the rule
every later rung inherits; I want its number on the board first.

Now the structural half, the larger and riskier one. In the default block the residual is updated twice in
series — the MLP reads the *post-attention* residual, depth 2 per block, 48 sequential sublayer writes across
the model. The parallel block instead reads *one* normalized copy of the pre-block residual and lets both
sublayers operate on it independently: `h = LN(x); x ← x + Attn(h) + MLP(h)`. The arithmetic is direct — one
shared norm per block instead of two, which on top of the RMSNorm saving takes the stack from 48 feature-axis
reductions per token (two RMS reductions × 24) down to 24 (one × 24). But I should be honest with myself
about whether "fewer reductions" is actually a *speed* win worth the name, because the numbers do not support
naive intuition here. A norm reduction touches `d = 1024` values per token; a single MLP projection is a
`1024 × 4096` matmul, `4,194,304` multiply-adds per token. One norm is therefore about `1024 / 4.19M ≈
0.024%` of *one* of the block's several large matmuls in raw arithmetic. Removing a norm saves essentially
*no FLOPs*. The reason a norm costs any wall-clock at all is memory bandwidth, not arithmetic: it is a
low-intensity kernel that streams the full `N×d` activation tensor (once to reduce, once to apply) at a
fraction of peak throughput, plus kernel-launch overhead. So dropping a norm saves a couple of tensor passes
and a launch, which is real but small. And the other advertised parallel win — the *shortened critical path*
from running attention and MLP concurrently — is only realized if the two branches actually fuse or overlap,
which in the large-scale recipes comes from fusing the attention and MLP *input projections* into one matmul.
That fusion lives inside `CausalSelfAttention` and `MLP`, which are frozen, outside my edit surface. I can
share the norm and remove the write-ordering dependency, but I cannot fuse the projections, so the two
matmul-heavy branches still launch back to back. My honest prediction, then, is that this block is the
*fastest* rung but only by the memory traffic of the one norm it drops — a few percent, not a halving. If a
later, non-parallel block that keeps this same RMSNorm comes in only slightly slower, that will confirm the
speed win was just the norm; if the parallel block is *dramatically* faster, I was wrong about the projections
not overlapping.

What parallelizing costs *representationally* is exactly why I expect this to be the weakest rung on quality,
and I want to be precise about the mechanism rather than leaning on "less cross-talk" as a slogan. The two
sublayers do different jobs: attention *mixes across positions* — it gathers, for each token, a
context-weighted sum of value vectors from other tokens — while the MLP *mixes across channels* at a fixed
position, a per-token nonlinear feature transform. In the sequential block the composition order is
token-mix-then-channel-mix *within the same layer*: the MLP at block ℓ transforms features that already
contain the context attention just gathered at block ℓ, so a single layer can express "gather this context,
then nonlinearly transform it." In the parallel block that composition is severed — the MLP at block ℓ
transforms the *pre-attention* `h`, and attention's freshly gathered context is only folded in at block ℓ+1,
one layer later. The model can still express the same composition, but it costs it a *layer of depth* to do
so, and with only 24 layers that is a non-trivial fraction of the budget to spend on re-integrating what a
sequential block did for free. That is the concrete content of "lost cross-talk," and the lineage is explicit
that it shows up as a *small quality loss at small scale* that vanishes only at very large scale — no
degradation reported at 62B, the loss visible below that — because at very large depth and width the one-layer
lag is a rounding error against the total capacity, while at 355M and 24 layers it is not. A 355M model is
firmly in the small-scale regime where the parallel approximation should bite. There is a second, subtler risk from
*combining* the two changes: with one shared RMSNorm and a summed residual `x + Attn(h) + MLP(h)`, the two
branch outputs add into the stream with no intervening normalization, and RMSNorm — unlike the `LayerNorm` I
am removing — does not recenter, so any drift in the mean of `h` is no longer cleaned up before it feeds two
branches at once. In a pre-norm stream whose variance already grows with depth, summing two un-normalized
branch outputs per block could let that growth run a little hotter. I do not have a number to tell me how much
hotter; part of what this rung exists to do is find out.

Before I commit I have to check the one way this could go badly wrong rather than merely suboptimally — the
initialization contract, because the scaffold's residual-scaling trick was tuned for the sequential block. The
fixed init scales every `c_proj` weight by `1/√(2·n_layer) = 1/√48 ≈ 0.1443` so that the variance each block
adds to the residual stream stays bounded; the `2` in the denominator is there because the sequential block
writes to the residual *twice* per block (once from attention's `c_proj`, once from the MLP's), for `2·24 =
48` writes total, and scaling each write's variance by `1/48` keeps the total added variance across the stack
at `O(1)` rather than growing 48-fold — the residual RMS ends up roughly `√2×` its input rather than `√48×`.
Now count the parallel block's writes: it sums `Attn(h) + MLP(h)`, which is *also two writes* into the
residual per block, `48` total, same as sequential. So the `1/√48` factor still matches the number of writes,
each write still gets scaled by `1/48` in variance, and the total added variance is still `O(1)`. The wiring
change does *not* silently break the variance calibration the substrate depends on. That check is the thing I
most needed to pass, and it passes by direct count: the risk here is the representational one — lost cross-talk
and a possibly hotter mean-drift — not an init blow-up.

So the first rung is settled and it is a clean fill of the two-region edit surface: replace `LayerNorm` with
an RMSNorm that keeps only the gain and ignores the `bias` argument, and replace `Block` with a single
shared-norm parallel block that sums the attention and MLP outputs into the residual in one step. I touch
nothing in `CONFIG_OVERRIDES`. The distilled module and the literal scaffold code are in the answer.

Now the falsifiable expectations, stated against the only anchor I have, the default fill this replaces.
RMSNorm alone I expect to be roughly quality-neutral — it removes the operation I argued was not load-bearing
and changes no parameters, so the validation loss should not move much on account of the rule. The parallel
wiring is where I expect the cost: at 355M, in the small-scale regime, the lost intra-block cross-talk should
show up as a *slightly higher* validation loss than a non-parallel RMSNorm block would give, and the summed
un-normalized branches may nudge it the same direction. The compensation is speed — but a *modest* speed win,
the memory traffic of one dropped norm, not a dramatic one, because the fused projection that carries the
large-scale critical-path win is outside my reach. So my prediction for this rung is: it trains stably to
completion; it is the cheapest per iteration but only by a few percent; and its validation loss is
*competitive but not the best* — at or above whatever the same norm in a less aggressively wired block would
achieve, because the parallel approximation pays a small-scale quality tax that keeping the two sublayers in
sequence does not. If it instead *matched* such a block, that would tell me the cross-talk I am giving up does
not matter at this scale and the speed is free.

It is worth being concrete about which of the five reported metrics should move and which should not, because
they read different things and I want a prediction I can actually check. `val_loss` is the in-distribution
FineWeb cross-entropy — the thing the tax should register on first and most directly, since it is the same
objective the model trained on and the cross-talk lost is quality left on that objective. The WikiText-2 and
LAMBADA perplexities are held-out language modeling; they read the same underlying quality on out-of-training
text, so I expect them to move the *same direction* as `val_loss` — if the parallel block is a slightly worse
language model, all three should agree, and if they disagree that is a signal something more interesting than a
uniform quality shift is happening. LAMBADA in particular rewards long-range last-word prediction, which leans
on attention's context gathering, so if the one-layer lag in folding attention into the MLP hurts anywhere it
could show a touch more there. The downstream accuracies, ARC-Easy and HellaSwag, are zero-shot and coarse —
multiple-choice scoring is a blunt instrument at 355M, where these tasks sit only modestly above chance — so I
expect them to be the *least* sensitive witnesses and would not read much into small moves either way. The
metric I will trust as the verdict is `val_loss`, with the two perplexities as corroboration.

I run this rung first precisely to measure that tax: the gap between this floor and a later, less
aggressively wired block that keeps the RMSNorm is exactly the price of going parallel, and I want that price on the
board before I build anything on top of it.

The causal chain in one breath: the default block pays for two sequential sublayers each behind a
two-reduction `LayerNorm`, so the most aggressive structural simplification the surface allows is to (a) swap
`LayerNorm` for RMSNorm — dropping the mean-subtraction that controls location (`var(a−μ)=var(a)`) but not
spread, keeping the re-scaling invariance and the self-regulating quadratic gradient (`∂L/∂W` invariant to
input scale, inverse to weight scale, `R·a = 0` confirming the Jacobian kills the radial direction) at one
reduction instead of two and with identical parameter counts, so the swap is a pure operation change with no
capacity change and no bias to lose since `bias=False` — and (b) collapse the two norms into one shared norm
feeding attention and MLP in parallel, summing both into the residual in a single update; this realizes only
the shared-norm and dropped-write-ordering part of the speed win (a few percent, memory traffic of one norm),
not the fused-projection part that is outside the edit surface, at the expected cost of a small-scale quality
tax from lost intra-block cross-talk and a possibly hotter residual from summing two un-normalized branches —
the `1/√48` init still matches the two residual writes per block so there is no init blow-up — so I expect the
fastest rung but not the lowest loss, and I run it first to price that tax for everything that follows.
