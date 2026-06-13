The sandwich rung came back and it settled the question I was unsure of — in the direction I half-expected
but with a margin small enough to be its own verdict. Validation loss 2.3104, against the parallel rung's
2.3112: the sandwich did beat the parallel floor, so the first part of the thesis held (restore the
cross-talk, control the branch outputs, get back some quality). But it beat it by *0.0008* — eight
ten-thousandths of a nat — and it paid for that sliver with the slowest run on the board: 21,661s training and
586s eval, against the parallel rung's 19,747s and 406s. Worse, two of the secondary witnesses went the
*wrong* way: WikiText-2 perplexity rose to 46.8 (from 45.98) and LAMBADA to 72.08 (from 70.96), and PIQA
dropped to 62.46 (from 64.42). So the headline metric improved by a whisper while the perplexities — which
read the same language-modeling quality on held-out text — actually got worse. That is the signature of a
change that is *insurance, not improvement*: the four-norm sandwich bounded the residual-stream variance, which
nudged the in-distribution validation loss a hair, but the extra normalization did not make the model a better
language model — if anything the heavier per-block normalization slightly over-constrained it, which the
perplexity rise on out-of-training text exposes. This is exactly the "soft-prior / variance-insurance"
outcome I flagged: at 24 layers the output norm is not a final-loss lever, it is a stabilizer the model did
not actually need, and I paid the most wall-clock on the ladder for it.

So now I have two structural experiments and their verdicts. The parallel rung *removed* normalization and
structure (one shared norm, summed branches) and came in worst on val_loss because it lost cross-talk at small
scale. The sandwich rung *added* normalization and structure (four norms, output control) and came in
marginally better on val_loss but worse on perplexity and slowest in time. Read together, they bracket the
answer: more block restructuring in either direction is not buying language-modeling quality at this scale —
the parallel simplification cost cross-talk, the sandwich elaboration cost generalization and speed, and
neither moved the needle more than a whisker on the metric that matters. The lesson is that I have been
spending my edit budget on the *wiring* when the wiring is not where the quality is. The one change that was
unambiguously safe across both rungs was the normalization *rule* — RMSNorm trained stably both times and I
never had a reason to doubt it. So the move that the data actually points to is to stop restructuring the block
and return to the *simplest* possible RMSNorm block: keep the cheaper, well-behaved normalization rule, drop
every structural complication, and put the two sublayers back in the plain sequential pre-norm arrangement the
default block used — just with RMSNorm instead of LayerNorm. Strip back to the minimal change from the
default, and let the architecture be exactly what the substrate was tuned for.

Let me re-derive why the plain sequential RMSNorm block should be the strongest of the three, grounding it in
the two failures I just measured. The default block is `x ← x + Attn(LN_pre(x))`, `x ← x + MLP(LN_pre(x))` —
one pre-norm per sublayer, sequential, clean identity residual path. Everything about this arrangement is
what the substrate's `1/√(2·n_layer)` residual init, its learning rate, and its schedule were chosen for: two
residual writes per block, a single norm on each sublayer's input, the gradient flowing straight down the
identity path without passing through a norm. The pre-norm placement gives the depth-stable, warmup-free
gradient that lets a 24-layer stack train cleanly; the sequential ordering gives the MLP the post-attention
residual to read, the intra-block cross-talk the parallel rung lost and which cost it 2.3112. So the *only*
thing I want to change from the default is the normalization rule, swapping LayerNorm for RMSNorm — and that
swap I have already validated twice as harmless to stability.

Why is the RMSNorm swap not just harmless but *right* here, beyond the arithmetic saving? The default
LayerNorm bundles two operations: subtract the mean (re-centering invariance — shift `a` by a constant, `μ`
absorbs it, output unchanged) and divide by the standard deviation (re-scaling invariance — scale `a` by `α`,
both `μ` and `σ` scale, the ratio is unchanged). Stabilizing a deep stack is about controlling the *spread* of
activations and gradients, and subtracting the mean does not touch the spread — `var(a − μ) = var(a)`; it
moves the cloud, it does not shrink or grow it. The operation that actually pins the magnitude the next block
and the backward pass see is the division by the scale. RMSNorm keeps exactly that — `āᵢ = aᵢ/RMS(a)·γ`,
`RMS(a) = √((1/n)Σaᵢ²)` — and discards only the re-centering, which I argue is along for the ride. When `a`
already has zero mean, RMSNorm and LayerNorm coincide exactly (`σ = RMS`), so this is the same layer with the
recentering switched off, not a different mechanism. And because the config sets `bias=False`, the default
LayerNorm carries no bias anyway, so there is nothing to lose there: a bias on a normalization layer only
exists to restore a location after recentering, and with no recentering there is no location to restore. The
RMSNorm therefore carries only the gain `γ`, initialized to ones, ignoring the `bias` argument it is handed.

The backward pass is the real reason I trust this as the substrate, and it is worth following because it
explains why RMSNorm should be *at least as good* as LayerNorm on quality, not merely cheaper. Because `RMS`
is quadratic in `a` and `a = Wx`, the weight enters both numerator and denominator. The Jacobian of the
normalized vector with respect to `a` is `R = (1/RMS(a))·(I − aaᵀ/(n·RMS(a)²))` — `1/RMS` times identity minus
a rank-one outer-product that projects out the radial direction (perturbing `a` along itself does not change
`a/RMS`, so the Jacobian must annihilate it; indeed `R·a = 0`). Chaining to the weight gives
`∂L/∂W = (R(γ ⊙ u))xᵀ` with `u` the upstream gradient. Scaling the input or the weights by `δ` sends
`R → R/δ`, so `∂L/∂W` is *invariant* to input scaling (the `δ` from `x` cancels the `1/δ` from `R`) and
*inversely proportional* to weight scaling (only `R` moves): a layer whose weights have grown large
automatically receives smaller gradients, an implicit per-layer learning-rate adaptation that damps further
growth with no schedule and no extra parameters. None of that self-regulation came from the mean-subtraction;
all of it survives in RMSNorm. So the cheaper layer is also the well-conditioned one. The `1/n` inside the
root (rather than plain L2 `‖a‖`) is kept deliberately: it normalizes per-coordinate rather than per-vector,
so the scheme behaves consistently across the 1024-dimensional feature vectors here and would across layers of
other widths — plain L2 normalization, which forces a unit sphere independent of `n`, does not transfer the
same way.

Now reconcile this with the two failures, because the plain block's claim to be strongest is *relative to the
restructured ones I measured*. Against the parallel rung: the plain block restores the sequential ordering, so
the MLP reads the post-attention residual and the intra-block cross-talk the parallel rung lost is back — that
is the gap that cost 2.3112, recovered by doing nothing more exotic than not parallelizing. Against the
sandwich rung: the plain block has *one* norm per sublayer, not two, so it does not impose the output-variance
constraint that I just watched over-constrain the model and push the perplexities up; it lets each sublayer
write its natural contribution into the residual, and at 24 layers — deep enough to train cleanly under
pre-norm but not so deep that the un-normalized residual growth actually hurts final loss — that freedom is
worth more than the sandwich's variance insurance. So the plain block sits at the sweet spot the two
restructured rungs straddled: it has the parallel rung's cross-talk back without its simplification, and it
has the default's clean pre-norm gradient without the sandwich's over-constraint. It is also the block the
`1/√(2·n_layer)` init and the substrate schedule were *literally tuned for*, two residual writes per block,
one input norm each — so unlike the restructured rungs it incurs no init-vs-wiring mismatch at all.

I should be honest about how large I expect the win to be, because the sandwich already taught me that these
differences are small. The plain RMSNorm block differs from the sandwich only in the *placement* of
normalization (input-only vs input-and-output) and from the parallel block in *ordering and norm count*. None
of these is a capacity change; the model has the same parameters either way (RMSNorm even has fewer, no
biases). So I am not expecting a dramatic drop — I am expecting the plain block to claim the *best* val_loss on
the ladder by a margin similar in size to the gaps I have already seen, a few thousandths of a nat, and,
crucially, to do it *while the perplexities go the right way* — because the thing that made the sandwich's
perplexity rise (extra per-block normalization over-constraining out-of-distribution text) is exactly what the
plain block removes. That perplexity reversal is the falsifiable claim I care about most: if the plain block
lowers val_loss but the perplexities do *not* improve over the sandwich, then the val_loss ordering is noise at
this scale and the honest conclusion is that norm placement does not matter here. I do not touch
`CONFIG_OVERRIDES`: the entire point of returning to the plain block is that it is the arrangement the
substrate's learning rate and schedule were chosen for, so changing them would undo the alignment I am
deliberately restoring.

So the strongest rung is the *least* elaborate fill of the edit surface: keep the RMSNorm class from rung 1
verbatim, and replace the `Block` with the plain sequential pre-norm block — `ln_1`, attention, residual add;
`ln_2`, MLP, residual add — the default block with LayerNorm swapped for RMSNorm and nothing else. The
distilled module and the literal scaffold code are in the answer.

The falsifiable expectations, against both prior rungs' real numbers. First and most confident: this should
post the *best* validation loss of the three — below the sandwich's 2.3104 and the parallel's 2.3112 — because
it has the cross-talk the parallel rung lost and lacks the over-constraint the sandwich added. Second, the one
I most want to see: the perplexities should *reverse the sandwich's regression* — WikiText-2 back below 46.8
(toward or under the parallel rung's 45.98) and LAMBADA back below 72.08 (toward or under 70.96) — because
removing the output norm removes exactly the constraint that pushed them up. Third, wall-clock should sit
*between* the two: two RMSNorms per block is cheaper than the sandwich's four (so faster than 21,661s) but
sequential ordering forgoes the parallel rung's shortened critical path (so slower than 19,747s) — it should
not be the fastest, and that is fine, because speed was never the objective. Fourth, downstream accuracy should
hold or tick up — ARC-Easy and HellaSwag in the same band, and PIQA recovering from the sandwich's 62.46
toward the parallel rung's 64.42, since whatever the sandwich's extra normalization cost in PIQA is removed
here. This is where the ladder ends: the strongest arrangement on this edit surface is the simplest one — the
cheaper RMSNorm rule, which was safe every time, in the plain sequential pre-norm block the substrate was
built for — and the two restructuring experiments earned their place by showing, in measured numbers, that
spending the edit budget on wiring in either direction (parallel simplification or sandwich elaboration) buys
nothing at 355M that the plain block does not already have.

The causal chain in one breath: the sandwich beat the parallel floor by only 0.0008 val_loss while *raising*
the perplexities (WikiText-2 46.8, LAMBADA 72.08) and running slowest — variance insurance, not improvement,
the over-constraint of four norms at 24 layers → so stop restructuring the block and keep only the one change
that was safe every time, the RMSNorm rule (drop the mean-subtraction that controls location not spread, keep
the re-scaling normalization and its self-regulating quadratic gradient, no bias since `bias=False` and
nothing to recenter) → put the two sublayers back in the plain sequential pre-norm block the substrate's
`1/√(2·n_layer)` init and schedule were tuned for, which restores the cross-talk the parallel rung lost and
removes the output norm that over-constrained the sandwich → expecting the best val_loss of the three *with*
the perplexities reversing the sandwich's regression, at a wall-clock between the two — the simplest fill of
the edit surface as the strongest, which is itself the finding.
