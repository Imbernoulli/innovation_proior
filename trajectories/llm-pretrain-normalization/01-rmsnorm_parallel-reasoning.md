I am standing at the default fill and I have to pick the first thing to change, knowing nothing yet about
how any of these moves actually land — there are no measured numbers in front of me. So I reason from the
structure of the scaffold and from what the substrate is paying for, and I start with the change that
attacks the largest, most obvious inefficiency in the block: the fact that the default `Block` runs two
sequential sublayers each fronted by its own full `LayerNorm`. That is four normalization reductions per
block per token if I count attention and MLP separately (two for the mean/variance pair inside attention's
`ln_1`, two for the MLP's `ln_2`), and the two sublayers are strictly sequential — the MLP cannot begin
until attention's residual has been added and re-normalized. Both of those are costs the architecture
imposes on every one of the 24 blocks, every token, every one of 12,030 iterations, and neither is
obviously buying me final-loss quality. So the first rung is the most aggressive *structural* simplification
the edit surface allows: collapse the two norms into one and run the two sublayers in parallel, and at the
same time replace the normalization rule itself with the cheaper RMSNorm. I want to be clear with myself
that this rung bundles *two* changes, because the leaderboard will only show me one number and I need to
know what that number is a verdict on.

Let me take the normalization rule first, because it is the cleaner of the two and it is shared by every
rung I will try after this one. The default `LayerNorm`, within a single token's feature vector
`a ∈ R^1024`, computes the mean `μ = (1/n)Σ aᵢ`, subtracts it, divides by the standard deviation
`σ = √((1/n)Σ(aᵢ−μ)²)`, and applies a learned per-channel gain `γ` and bias `β`. That is two distinct
operations welded together. Subtracting the mean buys *re-centering invariance*: shift every coordinate of
`a` by the same constant and the output is unchanged, because `μ` shifts by exactly that constant and the
centered vector `a − μ` does not move. Dividing by `σ` buys *re-scaling invariance*: multiply `a` by a
positive `α` and both `μ` and `σ` scale by `α`, so `(a−μ)/σ` is untouched. Two operations, two invariances,
cleanly separable. The question that decides whether I can drop the mean is: which of these invariances is
doing the work of keeping training well-conditioned? And the answer, mechanically, is the re-scaling one.
Stabilizing optimization is about controlling the *spread* of activations and gradients — keeping them from
blowing up or vanishing as signal propagates through 24 residual blocks. Subtracting the mean recenters the
cloud of activations but does nothing to its variance: `var(a − μ) = var(a)`. It tidies the location, it
does not control the scale. The thing that actually pins the magnitude that the next block and the backward
pass see is the division by the scale. So I bet the mean-subtraction is dispensable, and I replace `σ` —
which is *defined* through `μ` — with a measure of spread around the origin that references no mean at all,
the root mean square `RMS(a) = √((1/n)Σ aᵢ²)`. The forward rule becomes `āᵢ = aᵢ/RMS(a) · γ`. When the
mean of `a` happens to be zero this is *identical* to the default layer (`σ` collapses to `RMS`), so this is
not a wild departure — it is the same layer with the recentering switched off. Re-scaling invariance
survives because `RMS(αa) = α·RMS(a)` is linear, exactly the property the proof needs; only re-centering is
discarded, which is the invariance I argued does not matter.

Two consequences of dropping the mean matter for *this* scaffold specifically. First, the bias. The default
`LayerNorm` carries `β`, but the config sets `bias=False`, so in the actual default fill there is no bias
anyway — and that is the right call for a pre-norm transformer: the only reason to carry a per-channel shift
on a normalization layer is to restore a location after recentering, and RMSNorm does no recentering, so
there is nothing to restore. My RMSNorm therefore keeps only the gain `γ`, initialized to ones, and ignores
the `bias` argument it is contractually handed. Second, the gradient behaves well, which is the real test —
the point of a normalization layer is not its forward invariances but whether it keeps the *backward* pass
conditioned. Because `RMS` is quadratic in `a` and `a = Wx`, the weight enters both the numerator and the
denominator, and working the Jacobian through gives the clean result that `∂L/∂W` is invariant to scaling
the input and *inversely* proportional to scaling the weights — a layer whose weights have grown large
automatically receives smaller gradients, an implicit per-layer learning-rate adaptation that falls straight
out of the quadratic form and damps further growth with no schedule. None of that self-regulation came from
the mean I am throwing away; all of it comes from the re-scaling structure I keep. So RMSNorm is strictly the
part of `LayerNorm` that I can defend as load-bearing, at one reduction (sum of squares) instead of two and
with the subtraction pass gone. Across 24 blocks × 7 billion tokens that is a real arithmetic saving, and —
the part I actually care about for quality — it should not *cost* me anything, because the piece removed was
not controlling the spread. This is the change every later rung inherits; I want to know it is safe before I
build on it, which is one reason to put it on the first rung where I can see its number.

Now the structural change, which is the larger and riskier half. In the default block the residual stream
is updated twice: `x ← x + Attn(LN(x))`, then `x ← x + MLP(LN(x))`. The MLP reads the *post-attention*
residual, so the two sublayers are in series — depth 2 per block, 48 sequential sublayers across the model.
The parallel block instead reads *one* normalized copy of the pre-block residual and lets both sublayers
operate on it independently: `h = LN(x); x ← x + Attn(h) + MLP(h)`. The arithmetic payoff is direct — one
norm per block instead of two (halving the normalization cost on top of the RMSNorm saving), and the two
matmul-heavy sublayers can be computed from the same input without waiting on each other, which shortens the
critical path and is where the reported large-scale speedups come from. I want to be precise that this is
*this task's* parallel block and not the generic recipe. The edit collapses to a single shared `LayerNorm`
(now RMSNorm) feeding both branches and a single summed residual update; it does not, for example, fuse the
attention and MLP input projections into one matmul the way GPT-J's implementation does for its speed win,
because the fixed `CausalSelfAttention` and `MLP` modules are outside the edit surface and I cannot touch
their internals. So the *quality* consequences of going parallel are exactly in play here, while only part
of the *speed* win (the shared norm and the shortened dependency, not the fused projection) is realized. The
edit surface lets me change how the sublayers are wired, not how they are implemented inside.

What does parallelizing cost in representational terms, and why do I expect this to be the *weakest* of the
rungs I will try? In the sequential block, the MLP sees the residual *after* attention has written to it —
it can condition its computation on what attention just produced this layer. In the parallel block both
sublayers see the same stale `h` and cannot react to each other within the block; whatever cross-talk the
sequential ordering allowed is deferred to the next block. This is a genuine reduction in the per-block
expressive power, and the prior lineage is explicit that it shows up as a *small quality loss at small
scale* that vanishes only at very large scale (PaLM reports no degradation at 62B; the loss is visible
below that). A 355M model is firmly in the "small scale" regime where the parallel approximation is expected
to bite. So I am deliberately leading with the rung that trades the most quality for the most speed — it is
the natural floor of the ladder: maximally simplified structure, on a model small enough that the
simplification should cost something. There is a second, subtler risk specific to *combining* the two
changes. With one shared RMSNorm and a summed residual `x + Attn(h) + MLP(h)`, the two branch outputs add
directly into the stream with no intervening normalization, and RMSNorm — unlike the default LayerNorm I am
removing — does not recenter, so any drift in the mean of `h` is no longer being cleaned up before it feeds
two branches at once. In a pre-norm residual stream whose variance already grows with depth, summing two
un-normalized branch outputs per block could let that growth run a little hotter than the sequential block
did. I do not have a number to tell me how much; the rung exists partly to find out.

Let me also be honest about the initialization interaction, because the scaffold's residual-scaling trick
was tuned for the sequential block. The fixed init scales every `c_proj` weight by `1/√(2·n_layer)` so that
the variance added to the residual stream per block stays bounded — the `2` in that denominator is there
because the sequential block writes to the residual *twice* (once from attention's `c_proj`, once from the
MLP's). The parallel block also writes twice (it sums two branch outputs), so the `1/√(2·n_layer)` factor
still matches the number of residual writes per block, and I should not expect a gross variance
miscalibration from the wiring change alone. That is reassuring: the structural edit does not silently break
the init contract the substrate depends on. The risk is the representational one above, not an init blow-up.

So the first rung is settled and it is a clean fill of the edit surface: replace the `LayerNorm` class with
an RMSNorm that keeps only the gain and ignores the `bias` argument, and replace the `Block` class with a
single shared-norm parallel block that sums the attention and MLP outputs into the residual in one step. I
touch nothing in `CONFIG_OVERRIDES` — the learning rate and schedule the substrate fixes were chosen for
this model and I have no diagnostic yet that would justify moving them. The distilled module and the literal
scaffold code are in the answer.

Now reason about what this floor should do, because that is the whole point of running it first. RMSNorm by
itself I expect to be roughly neutral on quality — it removes the part of the norm I argued was not
load-bearing, so the validation loss should not move much on account of the norm rule. The parallel wiring
is where I expect the cost: at 355M, in the small-scale regime, the per-block loss of cross-talk between
attention and MLP should show up as a *slightly higher* validation loss than a sequential block would give,
and the summed un-normalized branches may let the residual variance grow a touch faster, nudging the loss
the same direction. The compensation is speed — fewer norms and a shorter critical path should make this the
*fastest* of the rungs in wall-clock. So my falsifiable expectation for this rung, against the only anchor I
have (the default fill it replaces), is: it should train stably to completion, it should be the cheapest per
iteration, and its validation loss should be *competitive but not the best* — specifically, I expect it to
land at or above whatever a sequential RMSNorm block achieves, because the parallel approximation is paying a
small-scale quality tax that the sequential block does not. If instead it matched or beat a sequential block,
that would tell me the cross-talk I am giving up does not matter at this scale, and the speed would be free.
The point of putting it first is to measure that tax directly: the next rung will keep the cheaper RMSNorm
but undo the parallel simplification and instead spend structure on *more* normalization, and the gap
between that rung and this one is exactly the price of going parallel that I am leaving on the table here.

The causal chain in one breath: the default block pays for two sequential sublayers each behind a full
LayerNorm, so the most aggressive structural simplification the surface allows is to (a) swap LayerNorm for
RMSNorm — dropping the mean-subtraction that controls location but not spread, keeping the re-scaling
invariance and the self-regulating quadratic gradient at one reduction instead of two, and dropping the bias
that only existed to undo recentering — and (b) collapse the two norms into one shared norm feeding attention
and MLP in parallel, summing both into the residual in a single update; this realizes the shared-norm and
shortened-critical-path part of the speed win (not GPT-J's fused projection, which is outside the edit
surface), at the expected cost of a small-scale quality tax from the lost intra-block cross-talk and a
slightly hotter residual variance from summing two un-normalized branches — so I expect the fastest rung but
not the lowest loss, and I am running it first precisely to price that tax for everything that follows.
