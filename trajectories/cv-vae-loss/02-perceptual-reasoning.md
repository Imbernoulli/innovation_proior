The L1 + KL floor told me exactly what is missing, and it told me in numbers that split hard across
scales. On the large model best rFID came in at 5.35 — already respectable — but the medium landed at
15.75 and the small at **53.00**, and the primary metric, the geometric mean across the three, is
`(53.00·15.75·5.35)^{1/3} = 16.47`. Before I react to that, let me actually read the spread rather than
just note it is large, because the *shape* of the split is the diagnosis. The three runs differ only in
capacity — the small has the narrowest channels and `4` latent channels through the `f=4` bottleneck, the
medium `8`, the large `16`, a clean doubling ladder — and the rFID falls `53.00 → 15.75 → 5.35` along it.
Take the ratios: `53.00/15.75 = 3.37` from small to medium, `15.75/5.35 = 2.94` from medium to large. So
each doubling of latent capacity buys roughly a *threefold* cut in rFID, and in log terms the drops
(`ln 3.37 = 1.21`, `ln 2.94 = 1.08`) are nearly equal — the rung is behaving like a straight line in
`log(rFID)` versus `log(capacity)`. That regularity is not noise; it is a mechanism. Where the model has
room to carry high-frequency detail (large), a pixel loss gets out of its way and the reconstruction is
crisp enough that even a feature-distribution metric like rFID is satisfied; where the model is starved
(small), the pixel loss does exactly what I feared — under reconstruction uncertainty it hedges toward
the conditional mean, the blurry average over all plausible textures, because that minimizes mean
absolute error. The `53.00` is the blur tax, and it is a smooth, capacity-indexed tax: largest precisely
where the bottleneck squeeze is tightest and the model can least afford to spend its scarce degrees of
freedom on pixel-mean-matching. The full small-to-large spread is `53.00/5.35 = 9.9×` — nearly an order
of magnitude of rFID separating the same loss at two capacities. This is not a learning-rate problem and
not a capacity problem I am allowed to touch; it is a *measurement* problem, and it is worst exactly
where measurement matters most. This lines up with the bottleneck geometry — the same `~3×`-per-halving
factor falls out of the compression ratios `D/J` = `12:1 → 6:1 → 3:1`, so squeeze and capacity are two
names for one thing, and the small model pays a `~9.9×` rFID penalty for a `4×` tighter bottleneck
because the detail it discards is exactly the high-frequency content a pixel loss under-rewards and rFID
over-weights.

So the obstacle is sharp: per-pixel L1 (and L2, and PSNR, which is L2 on a log axis) treats the image as
a bag of independent pixels, and that independence assumption is where it breaks. Blur a reconstruction
and the mean absolute residual can stay modest while the perceptual damage is obvious, because the
high-frequency structure that makes an image look sharp has been washed out and a pixel loss does not
know it should care. rFID, by contrast, compares the *distributions of deep features* between originals
and reconstructions — it is sensitive to exactly the softness a pixel loss is blind to. So I have an
objective (L1) that disagrees with my metric (rFID), and the small-scale `53.00` is what that
disagreement costs. The fix has to make the *training* loss see what rFID sees: I need to measure
reconstruction error somewhere that pixel softness is penalized, not in pixel space.

Where would such a measurement come from? The lever is a fact from image synthesis that is the same blur
story in another guise: when people train an image model under per-pixel L2, the outputs come out
over-smoothed, for exactly my reason — squared error is minimized by averaging over plausible
completions. The fix that swept the field was to stop measuring the error in pixel space and measure it
in the feature space of a pretrained classification network — VGG features. Push both the reconstruction
and the target through a fixed network and compare deep activations. The diagnostic that comes with this
is the part worth chewing on: if you optimize an image to match a target at an *early* VGG layer the
result is visually indistinguishable, while matching at a *deep* layer preserves content and layout but
not exact texture — so Euclidean distance *in these feature spaces already behaves perceptually*, and the
layers ladder from appearance to semantics. The network was never trained to do this; the perceptual
behavior is an emergent side effect of learning a representation predictive about natural images. That is
exactly the distance I want my small model to be scored on during training, because a blurred
reconstruction has visibly different deep features and would be penalized, whereas under L1 it slides
through.

The one real question is how much of that feature-space machinery to build myself, and the contract
settles it: `lpips` is an available *import* — a perceptual distance to *consume*, not one to learn.
Building my own calibrated metric is infeasible anyway with no 2AFC judgments in this loop, no second
optimizer for calibration coefficients, and a one-module edit surface rather than a metric-learning
pipeline. And a *raw* deep-feature distance — a plain sum of squared VGG-activation differences — fails
on a computable pathology: channels have wildly different activation magnitudes, so if channel A fires
at `~100` and channel B at `~1`, a `1%` error in the loud channel and a *complete* `100%` error in the
quiet one are both an absolute difference of `1`, scored identically. An unnormalized distance measures
whether the *loud* channels matched in gain, not whether the activation *pattern* agrees — and pattern
is where the perceptual content lives. LPIPS fixes exactly this: unit-normalize across channels at each
location (a gain-invariant cosine of feature *direction*), recalibrate each channel by a coefficient fit
to human judgments, spatially average, sum across layers.

So I take it cleanly: `lpips.LPIPS(net='vgg')`, which is precisely the whole object —
a frozen VGG trunk, channel normalization, learned per-channel calibration grounded in human data —
pre-built and pre-calibrated. I freeze it hard: `eval()` and `requires_grad_(False)` on every parameter.
Freezing is not optional, and the reason is the same emergent-perceptual-structure fact I am exploiting.
If I let gradients into the VGG trunk I would be retraining the very representation whose accidental
perceptual goodness is the thing that makes this distance work, and I would bend it toward this one
dataset — dissolving the property I imported it for. Conservative read: calibrate nothing, retrain
nothing, just measure with it. `eval()` also matters mechanically because VGG carries no batchnorm
running-stat surprises in eval, and `requires_grad_(False)` keeps its parameters out of the optimizer and
its activations from holding a graph I never backprop into its weights.

The `net='vgg'` choice inside LPIPS is deliberate rather than a default I am accepting blindly. LPIPS
ships several trunks — a lighter AlexNet variant, a SqueezeNet variant, and VGG — and they trade off
differently depending on whether the distance is used to *score* or to *train*. As a raw evaluation
number the lighter trunks can track human judgment about as well, but I am not scoring with this; I am
putting it inside the loss and backpropagating through it every step, so what I need is a distance whose
*gradient* is smooth and dense enough to steer the decoder. VGG's stack of small convolutions gives a
gradient field that is less spiky than the aggressively strided lighter trunks, which is the property that
matters when the signal has to flow back through the whole decoder toward the encoder. So I take the VGG
trunk specifically for its behavior as a training loss, not merely as a metric.

I should also be honest about what this term is and is not with respect to the metric I am scored on. The
scored rFID is a Fréchet distance between *Inception* feature distributions; LPIPS is a *VGG* feature
distance. They are not the same network and not the same statistic, so I am optimizing a *correlated
proxy* for rFID, not rFID itself. The correlation is strong — both are deep features of natural-image
classifiers, and blur that shifts one tends to shift the other — but it is not perfect, and the honest
risk is that the decoder finds a way to please VGG that Inception's distribution distance does not fully
credit. I accept that risk because I have no differentiable handle on Inception here and VGG is the best
aligned, backpropable stand-in available; the rFID column will tell me how well the proxy tracked, and if
a scale improves on LPIPS-driven training but its rFID barely moves, that decoupling is the thing to
suspect.

Now compose. I do not throw away the floor — the negative-ELBO skeleton was correct, the KL is still the
right closed-form leash at the same tiny `1e-6`, and I keep the L1 reconstruction term, because a
distance defined purely on deep features has a large null space: many pixel-level images collapse to
nearly the same VGG activations, so an optimizer pushed only by LPIPS can drift into that null space and
paint high-frequency junk (checkerboard or grid-like texture VGG fails to distinguish from the target)
while the feature distance keeps falling. L1 has no such null space at the pixel level; every wrong pixel
costs. So the two terms are complementary by construction — L1 rules out the pixel-space nonsense LPIPS
is blind to, LPIPS rules out the perceptual softness L1 is blind to. The reconstruction objective becomes
`L1(recon, target) + perceptual_weight · LPIPS(recon, target)`, plus `kl_weight · KL`. Enlarging the
reconstruction side does not make a `1e-6` KL suddenly steer — if anything its share *falls* below the
floor's `~0.7%` — so I leave `kl_weight = 1e-6` where it was.

I should be clear-eyed that this move costs me the clean probabilistic story. The floor was an honest
negative ELBO — every term a log-likelihood or a divergence. Once I add `0.5·LPIPS`, the objective is no
longer a bound on any marginal likelihood; it is an ELBO plus a perceptual regularizer with no
generative-model interpretation. That is a deliberate trade: the task scores rFID, not log-likelihood,
and a term that is a poorly-motivated likelihood but a well-motivated proxy for the scored metric is the
right thing to optimize *here*. If I were being scored on likelihood I would not add it.

One implementation detail the harness forces, and it is a real numerical trap rather than a formality:
the loop runs under autocast mixed precision, and LPIPS is a VGG forward pass. Half precision carries
roughly ten mantissa bits — about three decimal digits — and VGG's deep activations have a large dynamic
range. The perceptual signal I care about lives in the *small differences* between the reconstruction's
features and the target's, and once the reconstruction is good those features are nearly equal, so
subtracting them is catastrophic cancellation: in fp16 the leading digits agree and the difference is
computed from the few bits of noise that survive. That is precisely the regime where the metric is
supposed to be most informative — a nearly-right reconstruction — and precisely where fp16 would corrupt
it. So I cast both inputs to float before the perceptual call — `self.lpips_fn(recon.float(),
target.float())` — to keep the feature extractor in the precision it was trained in and let the small
feature differences be resolved; the `.mean()` reduces its per-image output to a scalar.

The one real knob is `perceptual_weight`, and I set it by a units argument rather than a vibe. The two
terms are in different scales: `L1` on pixels in `[-1, 1]` is order `0.05–0.1` for a decent
reconstruction, while a calibrated LPIPS distance for a perceptually-close pair is order `0.1–0.5`. So
LPIPS is *already* naturally a few times larger than L1 before any weighting. If I set the weight to
something like `5`, the perceptual term would swamp L1 by an order of magnitude and I would lose the
coarse-structure anchor I just argued I need; if I set it to `0.05`, `0.05·LPIPS ~ 0.01` would sit an
order below L1 and barely shape the output. A weight around `0.5` lands `0.5·LPIPS ~ 0.05–0.25` in the
same order as L1 — large enough that LPIPS materially drives sharpness, small enough that L1 still
anchors coarse structure and color. That order-matching is the whole justification, and `0.5` is the
standard balance for this pairing for exactly this reason. I keep `kl_weight = 1e-6` unchanged, because
the floor's KL behavior was not the problem — the reconstruction *distance* was — and there is no reason
to disturb a leash that is doing its quiet job. So the literal edit is: in `__init__`, build and freeze
`lpips.LPIPS(net='vgg').to(device)`, set `perceptual_weight = 0.5` and `kl_weight = 1e-6`; in `forward`,
compute L1, the float-cast mean LPIPS, and the mean KL, sum them as `rec + 0.5·p_loss + 1e-6·kl`, and
report all three in the metrics dict. No discriminator, no frequency term yet — just the perceptual
upgrade to the reconstruction distance. (The distilled module is in the answer.)

Now the falsifiable expectations against the floor's numbers. The diagnosis was that the blur tax is
largest where capacity is tightest (`53.00` small, `15.75` medium, `5.35` large). The perceptual term
attacks blur directly — the one thing L1 was blind to — so if the diagnosis holds, the *biggest*
improvement should land on the *small* scale, where L1 left the most on the table; the large scale was
already near-crisp, so the perceptual term has less blur to remove and I expect a smaller absolute gain.
The crisp test: if instead the large scale improved most and the small barely moved, the whole "blur tax
scales inversely with capacity" reading would be wrong and I would rethink the mechanism from scratch.
PSNR and SSIM I expect to hold or dip very slightly: those reward pixel and
local-statistic agreement, which the pure L1 floor was already optimizing, and adding a perceptual term
that tolerates small pixel-exact misses in exchange for crispness can trade a hair of PSNR for a large
rFID gain — which is exactly the trade I want, since rFID is the scored metric and PSNR is only a
diagnostic. If PSNR *collapsed*, that would tell me `0.5` is too high and LPIPS is overriding the anchor;
holding roughly flat is the signature of a healthy balance.

And I can already feel where this rung will run out. The perceptual loss pulls deep features together,
but it has no sharp, adversarial opinion about whether a *patch* of texture is realistic — VGG features
can be matched by something close-but-still-a-bit-soft, because the metric is a fixed distance, not an
active critic hunting the residual difference between a real image and a reconstruction. So I expect
perceptual to substantially close the small-scale gap but leave a residual softness — the kind a fixed
distance structurally cannot finish — as the live constraint on top of this same L1 + LPIPS + KL
skeleton. The metrics dict reports `rec_loss`, `p_loss`, and `kl_loss` separately so I can watch which
term still binds when this rung plateaus.
