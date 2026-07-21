I'm at 12.0 epochs and 4.6 seconds. Lookahead took me 13.5 → 12.0 epochs and 5.1 → 4.6 seconds — 1.5
epochs, the smallest gain of any change so far, exactly what I predicted for a device that harvests only the
*residual* noise of fast SGD rather than removing gross wasted work. And per-epoch cost ticked from
5.1/13.5 = 0.378 s to 4.6/12.0 = 0.383 s — a ~1.3% *rise*, not a fall: the one thing Lookahead is not is
perfectly free, since the `lerp` every 5 steps costs a hair, paid back many times by the 1.5 epochs
removed. Initialization and optimization-rate are largely wrung out, and each new epoch is now cheaper to
run but harder to remove.

Every change so far worked on *training*. There is a second place to spend effort I have treated as fixed:
*inference*. The timer runs from first data access to emitting test predictions, and arbitrary test-time
augmentation is allowed. So inference cost counts against me — but inference *quality* can be bought, and
that is leverage I have not used. If I squeeze more accuracy out of the *same* trained network at
evaluation, the network need not be trained as far to clear 94%: I trade epochs of training (expensive,
paid over all 50,000 images every epoch) for a few extra forward passes at eval (cheap, paid over the
10,000 test images once).

Price the trade, because it can go negative. The baseline already does a little TTA: horizontal-flip,
evaluating each test image as-is and mirrored and averaging the two logits — 2 forward passes over 10k.
Go to 6 views and that is 4 *extra* forwards over 10k = 40,000 additional image-forwards, once. A
training epoch is 50,000 images each taking a forward *and* a backward, and a backward is roughly twice a
forward, so an epoch costs about 50,000 × 3 = 150,000 forward-equivalents. The 40,000 extra eval-forwards
are worth ~40,000/150,000 ≈ 0.27 of a training epoch. That is the break-even: if the richer inference
lets me cut *more* than about a quarter-epoch of training to hold 94%, the trade is net positive. A
quarter-epoch is a tiny threshold and TTA routinely buys more accuracy than that, so the trade looks
strongly favorable — but it also warns me the view count is not free to inflate without bound.

*Which* extra views? The right ones are the label-preserving transformations the network is nearly — but
not exactly — invariant to, because those are where averaging cancels real nuisance variance. The
network's prediction on a single view is a noisy estimate of the true posterior; averaging its
predictions over several label-preserving views cancels the residual sensitivity — the same
variance-reduction logic as the Lookahead weight averaging, applied over *input views* instead of *weight
iterates*. Flip already gives two views and captures the biggest nuisance (left/right). The next one is
told by the training augmentation: I trained with horizontal flip and small random *translations*, so the
network has been taught to be approximately invariant to small shifts — and "approximately" is exactly
the regime TTA helps. The discipline here is strict: TTA can only average over invariances the network
*actually has*, which for a trained network means transformations it *saw during training*. Rotations,
scalings, colour jitter, larger crops — the network trained under none of these, so feeding it those at
test time hands it out-of-distribution inputs it scores badly and inconsistently, and averaging a bad
view injects bias rather than cancelling variance. So the only safe views are flip and small translation.

Concretely: the image, plus a version shifted one pixel up-left and one shifted one pixel down-right —
three translation states, each in two flip states, six views. A translation at the boundary has to invent
pixels, so I reflection-pad the 32×32 image to 34×34 and crop two 32×32 windows: `[0:32, 0:32]` and
`[2:34, 2:34]`, offset by 2, so each differs from the original by one pixel and from each other by two —
matching the 2-pixel-scale translation the network trained under. Reflection padding rather than zero
keeps the invented border pixels statistically consistent with the image edge, so a translated view is a
plausible image, not one with a black seam.

The six are not equally trustworthy, so I do not average them uniformly. The *untranslated* views are the
actual test image, no invented pixels; the four translated ones are slightly degraded — a one-pixel shift
cropped from the padded image replaces one edge row and column, roughly 64 invented pixels out of 1024,
about 6% synthetic. So I weight the two untranslated views 0.25 each and the four translated 0.125 each
(2·0.25 + 4·0.125 = 1.0), giving the clean image half the total weight and down-weighting each translated
view 2:1 — the right order for ~6% corruption. I realize this as a *hierarchy* rather than six explicit
weights: an inner `infer_mirror` averages a view with its flip (0.5 each), and an outer function blends
the untranslated mirror-result 50/50 with the mean of the two translated mirror-results (code in the
answer). Tracing the weight onto each leaf confirms the hierarchy *is* the weighting, not an
approximation of it: an untranslated view enters at 0.5 in its flip pair × 0.5 in the final blend = 0.25;
a translated view at 0.5 × 0.5 (one of two translations) × 0.5 = 0.125.

I average the *logits*, not the post-softmax probabilities, and that is deliberate: averaging logits then
taking the softmax is a *geometric* mean of the per-view distributions (a product-of-experts), which
requires the views to agree before the ensemble is confident, so an idiosyncratic overconfident view
(an artifact of its crop) gets tempered by the others. Averaging probabilities would be an arithmetic
mean a single overconfident view can dominate.

How much variance this cancels bears on whether six views is enough. For a weighted average the effective
number of independent samples is 1/Σwᵢ² = 1/(2·0.25² + 4·0.125²) = 1/0.1875 ≈ 5.33 — close to the 6 raw
views, so the down-weighting of the noisier translated views costs little effective count. If the six
views' nuisance errors were independent that would cut nuisance variance ~5.3×; they are not (same image
lightly transformed, correlated errors), so the real reduction is smaller, but the ceiling is a ~5× cut.
The baseline already ran the flip pair (~2 effective views), so the *incremental* accuracy here is purely
the translation dimension — the step from ~2 to ~5.3. If the network is already quite translation-
invariant the realized cut stays near the flip-only ~2 and the win is small; if there is real residual
translation-sensitivity the cut approaches the ceiling. So this change's size is, quite directly, a
measurement of how much translation-sensitivity the training left in the network — which is why I can
predict its sign but not its magnitude.

In wall-clock terms the six-view evaluation over the 10k test set (in five chunks of 2000) is 30 forward
passes = 60,000 image-forwards, ~0.4 of one training epoch, of which the *extra* four views beyond the
flip pair are ~0.27 epochs, matching the price above. In a ~12-epoch run that added inference is well
under half an epoch. That diminishing-returns shape is why I stop at six and do not chase the classic
ImageNet regime of tens or 144 crops: flip captured the biggest nuisance, the one-pixel translations the
second, and after that each view buys less accuracy at the same fixed ~0.07-epoch eval cost per view, so
beyond six the trade turns.

TTA composes cleanly with everything before it: it changes nothing about the trained weights — a pure
post-hoc readout — so it cannot interfere with whitening, Dirac, the boosted biases, or Lookahead, it
just reads their network more carefully. And it exploits the cheapest compute in the pipeline (eval
touches 10,000 images once, forward-only, no gradient) to avoid spending the most expensive (training
epochs over 50,000 images with a backward each) — exactly the arbitrage the timer's rules permit. The
invariances I average over are themselves a *product* of the training augmentation, so this collects
at eval time the invariance training paid for.

So I expect epochs-to-94% to drop below 12.0, and the seconds to follow even after paying for six-view
inference, because the ~0.27-epoch eval cost is much smaller than the training epochs the accuracy boost
should save. The size should be modest — flip already captured the biggest nuisance and I am adding the
second with sharply diminishing returns. The honest failure mode is the trade going the wrong way: if the
network is *already* nearly translation-invariant, the translated views add little while still costing
their forwards, and it comes out close to a wash — epochs barely moving while seconds even tick up
from the eval cost. The code is in the answer.
