I'm at 12.0 epochs and 4.6 seconds, and reading the Lookahead win tells me two useful things before I
pick the next lever. Lookahead took me 13.5 → 12.0 epochs and 5.1 → 4.6 seconds: 1.5 epochs, the
smallest step of any rung so far, which is exactly what I predicted for a device that only harvests the
*residual* noise of fast SGD rather than removing gross wasted work. And the per-epoch cost ticked from
5.1/13.5 = 0.378 s to 4.6/12.0 = 0.383 s — a ~1.3% rise, not a fall. That small rise is the one thing
Lookahead is not: perfectly free. The `lerp` every 5 steps costs a hair, and it shows up here as a
slightly more expensive epoch, paid back many times over by the 1.5 epochs removed. So the ledger is
clean and the diagnosis held: initialization and optimization-rate are largely wrung out, and each new
epoch is now cheaper to *run* but harder to *remove*.

Every rung so far has worked on *training* — the initialization, the optimizer, the learning rates. There
is a second place to spend effort that I have been treating as fixed: *inference*. The rule of the
speedrun is that the timer runs from first data access to emitting test predictions, and arbitrary
test-time augmentation is allowed. So inference cost counts against me — but inference *quality* can be
bought, and that is the leverage I have not used. If I can squeeze more accuracy out of the *same* trained
network at evaluation time, then the network does not have to be trained as far to clear 94%: I can
shorten the training and let a smarter inference make up the last fraction of a percent. The trade is
epochs of training (expensive, and paid over all 50,000 images every epoch) for a few extra forward passes
at eval (cheap, and paid over the 10,000 test images exactly once).

Let me price that trade before I design anything, because it is the whole justification and it can go
negative. The baseline already does a little TTA: horizontal-flip, evaluating each test image both as-is
and mirrored and averaging the two logit vectors — 2 forward passes over the 10k test set. Suppose I go to
6 views. That is 4 *extra* forward passes over 10k = 40,000 additional image-forwards, once. A training
epoch, by contrast, is 50,000 images each taking a forward *and* a backward, and a backward is roughly
twice a forward, so an epoch costs about 50,000 × 3 = 150,000 forward-equivalents. So the 40,000 extra
eval-forwards are worth about 40,000 / 150,000 ≈ 0.27 of a training epoch. That is the break-even: if the
richer inference lets me cut *more* than about a quarter-epoch of training to hold 94%, the trade is net
positive, and every additional epoch it saves beyond that is pure profit. Since a quarter-epoch is a tiny
threshold and TTA routinely buys more accuracy than that, the trade looks strongly favorable — but pricing
it this way also warns me that the view count is not free to inflate without bound, which I will return
to.

Now, *which* extra views? The right ones are the label-preserving transformations the network is nearly —
but not exactly — invariant to, because those are where averaging can cancel real nuisance variance. The
network's prediction on a single view is itself a noisy estimate of the true class posterior: the network
is slightly sensitive to transformations it *should* be invariant to, and averaging its predictions over
several label-preserving views of the same image cancels that sensitivity. This is the identical
variance-reduction logic as the Lookahead averaging — the mean of noisy estimates has lower variance than
any one — but applied over *input views* instead of *weight iterates*. Flip already gives me two views and
captures the biggest nuisance (left/right). What is the next one? The training augmentation tells me
exactly: I trained with horizontal flip and small random *translations*. So the network has been taught to
be approximately invariant to small shifts — and "approximately" is precisely the regime TTA helps, where
there is residual sensitivity to average away. So the views to add are small translations, matched to the
augmentation the network already half-learned to ignore.

I should be disciplined about which transformations qualify, because the temptation is to throw more kinds
of augmentation at eval and it would backfire. The rule that makes TTA help is that the transformation
must be one the network is *approximately invariant* to — which, for a trained network, means one it *saw
during training*. Rotations, scalings, colour jitter, larger crops: the network trained under none of
these, so it is not invariant to them, and feeding it a rotated or recoloured test image is handing it an
out-of-distribution input it will score badly and inconsistently. Averaging in a bad view does not cancel
nuisance variance; it injects bias. So the only safe TTA views are the ones whose transformation matches
the training augmentation distribution — flip and small translation, nothing else. That is not a
conservative choice, it is the correct one: the training augmentation defines the invariance the network
actually has, and TTA can only average over invariances the network actually has.

Concretely: take the image, and also a version shifted one pixel up-and-to-the-left and one shifted one
pixel down-and-to-the-right. Three translation states (none, up-left, down-right), each in two flip states
(as-is, mirrored), gives six views of each test image. Let me pin down the mechanics of producing the
shifts, because a translation at the image boundary has to invent pixels. I reflection-pad the 32×32 image
by one pixel to 34×34, then crop two different 32×32 windows out of it: the top-left window `[0:32, 0:32]`
is the image shifted one pixel down-and-right in content terms (equivalently the field of view moved
up-left), and the window `[2:34, 2:34]` is the opposite shift. Reflection padding rather than zero padding
keeps the invented border pixels statistically consistent with the image edge, so the translated view is a
plausible image rather than one with a black seam. The shape bookkeeping checks: pad 1 on each side of 32
gives 34; `[0:32]` and `[2:34]` are both length-32 windows, offset by 2, so the two translated views
differ by a 2-pixel diagonal shift and each differs from the original by 1 pixel. That matches the
2-pixel-scale translation the network trained under.

How to combine the six matters, because they are not equally trustworthy. The *untranslated* image is the
cleanest view — it is the actual test image, no invented pixels — while the four translated ones are
slightly degraded by the reflection-padded border. So I should not average all six uniformly; I should
weight the untranslated views more. The scheme I want: the two untranslated views (as-is + mirrored) get
weight 0.25 each, and the four translated views (two translations × two flips) split the remaining half at
0.125 each. Let me check the weights sum to one: 2 × 0.25 + 4 × 0.125 = 0.5 + 0.5 = 1.0. Good — it is a
proper weighted average, and the clean untranslated image carries half the total weight on its own, which
is what I want. The 2:1 down-weighting of the translated views (0.25 vs 0.125) is not arbitrary either: a
one-pixel shift cropped from the reflection-padded image replaces one edge row and one edge column with
reflected pixels, roughly 2 × 32 = 64 invented pixels out of 1024, about 6% of the image. So each
translated view is a genuine but small corruption of the input — ~6% synthetic pixels — and halving its
weight relative to the pristine untranslated view is the right order for how much less I should trust it.
The untranslated views anchor the ensemble; the translated views act as smaller corrections.

I realize the average as a *hierarchy* rather than as six explicit weights, and I should verify the
hierarchy reproduces exactly the weights I just specified, or the code and the intent would silently
diverge. The inner function `infer_mirror` averages a view with its horizontal flip, 0.5 each. The outer
function `infer_mirror_translate` computes `infer_mirror` on the untranslated image and on each of the two
translations, then blends the untranslated result and the *mean* of the two translated results 50/50.
Trace the weight that lands on each of the six leaf views. An untranslated view: it enters through the
untranslated `infer_mirror` (weight 0.5 within its flip pair) which is then given weight 0.5 in the final
blend → 0.5 × 0.5 = 0.25. A translated view: it enters through its translation's `infer_mirror` (0.5
within its flip pair), then that translation is one of two averaged into the translated mean (× 0.5), then
the translated mean gets 0.5 in the final blend → 0.5 × 0.5 × 0.5 = 0.125. So the hierarchy yields 0.25
for each untranslated view and 0.125 for each translated view — exactly the target weights. The nested
averages are not an approximation of the weighting; they *are* it.

```python
def infer_mirror(inputs, net):
    return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

def infer_mirror_translate(inputs, net):
    logits = infer_mirror(inputs, net)
    pad = 1
    padded_inputs = F.pad(inputs, (pad,)*4, "reflect")
    inputs_translate_list = [
        padded_inputs[:, :, 0:32, 0:32],   # up-and-to-the-left by one pixel
        padded_inputs[:, :, 2:34, 2:34],   # down-and-to-the-right by one pixel
    ]
    logits_translate_list = [infer_mirror(t, net) for t in inputs_translate_list]
    logits_translate = torch.stack(logits_translate_list).mean(0)
    return 0.5 * logits + 0.5 * logits_translate
```

One detail in the combination that is easy to get wrong: I average the *logits* — the raw `net(inputs)`
outputs — not the post-softmax probabilities. That is deliberate. Averaging logits and then taking the
softmax is equivalent to taking a *geometric* mean of the per-view probability distributions (a
product-of-experts combination), which requires all the views to agree before the ensemble is confident
and is the standard, well-behaved way to pool a network's predictions across augmentations. Averaging
probabilities instead would be an arithmetic mean, which a single overconfident view can dominate. Since
the six views are the same image lightly perturbed, I want the "all must agree" behavior of the logit
average — a view that is confident for an idiosyncratic reason (an artifact of its particular crop) gets
tempered by the others rather than allowed to carry the vote. So logit-space averaging, throughout the
hierarchy.

Let me also quantify how much variance this actually cancels, because the answer bears on whether six
views is enough. For a weighted average of views, the effective number of independent samples is 1 / Σ
wᵢ², the inverse participation ratio of the weights. Here Σ wᵢ² = 2 × 0.25² + 4 × 0.125² = 2 × 0.0625 + 4 ×
0.015625 = 0.125 + 0.0625 = 0.1875, so the effective sample size is 1 / 0.1875 ≈ 5.33 — meaning the
weighting is efficient (close to the 6 raw views; the down-weighting of the noisier translated views costs
only a little effective count). If the six views' nuisance errors were independent, that would cut the
nuisance variance by ~5.3× — a ~2.3× reduction in nuisance standard deviation. The views are not
independent (they are the same image lightly transformed, so their errors are correlated), so the real
reduction is smaller, but the calculation tells me the *ceiling* is a ~5× variance cut and that the
weighting is not wasting effective views on the noisy translations. It also frames the diminishing returns:
going from 1 effective view (no TTA) to ~2 (flip) captures the first big drop, and ~2 to ~5.3 captures a
smaller second drop. And it sharpens what this rung actually measures: the baseline already ran the flip
pair (level 1), so the *incremental* accuracy I am buying here is purely the translation dimension — the
step from ~2 effective views to ~5.3. If the correlation between the translated views and the original is
high (the network being already quite translation-invariant), the realized variance cut will be well short
of the 5.3× ceiling and closer to the flip-only ~2, and the incremental win will be small; if there is
real residual translation-sensitivity, the cut approaches the ceiling and the win is larger. So the size
of this rung is, quite directly, a measurement of how much translation-sensitivity the training left in
the network — which is why I cannot predict its magnitude precisely in advance, only its sign.

It helps to put the eval cost in wall-clock terms, since that is what the timer charges me. The inference
runs over the 10k test set in chunks — `test_images.split(2000)` gives five batches of 2000 — and each
chunk runs the six-view `infer_fn`, so the whole evaluation is 5 × 6 = 30 forward passes of 2000 images,
30 × 2000 = 60,000 image-forwards total, once. Against a training budget where a single epoch is 150,000
forward-equivalents (50k images, forward plus a ~2× backward), the entire six-view evaluation is about
60,000 / 150,000 ≈ 0.4 of one training epoch of compute — and the *extra* four views beyond the
already-present flip pair are two-thirds of that, ~0.27 epochs, matching the price I quoted up front. So
in a run that is now ~12 epochs long, the added inference is well under half an epoch of time. That is the
quantitative reason the trade can be favorable: I am spending a fraction of one epoch at eval to try to
save a whole epoch or more of training.

```python
def infer(model, loader, tta_level=0):
    model.eval()
    test_images = loader.normalize(loader.images)
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]   # level 2 = multi-crop
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])
```

The `tta_level` index is a clean dial: level 0 is a single view, level 1 is the flip pair the baseline
used, level 2 is the six-view multi-crop I am adding — so the change is entirely opt-in and the cheaper
levels remain available for comparison. The `split(2000)` is just there to keep six copies of a
2000-image batch's activations within memory; it does not change the math, only the granularity of the
forward passes.

That diminishing-returns shape is exactly why I stop at six and do not chase the classic ImageNet
multi-crop regime, where people go to tens or even 144 crops and accuracy does keep inching up. Each extra
view is another full forward pass over the 10k test set, and I priced those at ~0.27 epochs per four
views; pushing to dozens of crops would cost several epochs of equivalent eval time to claw back a
fraction of a percent that the effective-sample-size curve says is flattening hard. Flip captured the
biggest nuisance (left/right), the one-pixel translations capture the second-biggest (small shift), and
after that each view is buying less accuracy at the same fixed eval cost. Six views is where the marginal
accuracy still clearly beats the marginal 0.07-epoch eval cost of one more; beyond that the trade turns.
So: six views, weighted, and stop.

There is a structural reason this lever is worth pulling now and composes cleanly with everything before
it. TTA changes *nothing* about the trained weights — it is a pure post-hoc readout, run after training
finishes — so it cannot interfere with whitening, Dirac, the boosted biases, or Lookahead; it sits
downstream of all of them and simply reads their trained network more carefully. And it exploits the
cheapest compute in the whole pipeline: training touches all 50,000 images every epoch and pays a backward
pass on each, while evaluation touches 10,000 images once, forward-only, with no augmentation bookkeeping
and no gradient. So a forward-eval-pass is intrinsically the least expensive unit of work available to me,
and buying accuracy with the cheapest possible currency to avoid spending the most expensive one
(training epochs) is exactly the arbitrage the timer's rules permit. The approximate invariances I am
averaging over are themselves a *product* of the training augmentation — the flip-and-translate the
network half-learned to ignore — so this rung is, in a sense, collecting at eval time the invariance that
training paid for.

The prediction, made falsifiable against the epoch column. The mechanism is real and it is the same one I
just used on weights: averaging the network over label-preserving views it is only approximately invariant
to should shave the test error, and — crucially — I can cash that error reduction as *fewer training
epochs* to reach 94%, because a network that scores a little higher at eval needs to be trained a little
less to clear the bar. So I expect epochs-to-94% to drop below 12.0, and the seconds to follow it down
even after paying for six-view inference, because the ~0.27-epoch eval cost is much smaller than the
training epochs the accuracy boost should save. The size should be modest — flip-TTA was already capturing
the biggest nuisance, and I am adding the second-biggest with sharply diminishing returns per the
effective-sample-size curve. The honest failure mode is the trade going the wrong way: if the network is
*already* nearly translation-invariant (the training translations having taught it well), the translated
views add little signal while still costing their forward passes, and the rung could come out close to a
wash — I would see the epoch count barely move while the seconds even tick up slightly from the eval cost.
But six cheap views against a saved epoch of training is a favorable bet on the numbers above. The change
is the `tta_level=2` multi-crop inference path; the code is in the answer, and the epochs table will tell
me whether the network still had translation-sensitivity left to average away.
