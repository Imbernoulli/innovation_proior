The L1 + KL floor told me exactly what is missing, and it told me in numbers that split hard across
scales. On the large model best rFID came in at 5.35 — already respectable — but the medium landed at
15.75 and the small at **53.00**, a tenfold spread for the same loss. That spread is the diagnosis. The
loss is per-pixel L1, and the only thing that changes across the three runs is capacity: the small model
has the narrowest channels and just 4 latent channels through the `f=4` bottleneck, the large has the
widest channels and 16. So where the model has room to carry high-frequency detail (large), a pixel loss
gets out of its way and the reconstruction is crisp enough that even a feature-distribution metric like
rFID is satisfied; where the model is starved (small), the pixel loss does exactly what I feared it
would — under reconstruction uncertainty it hedges toward the conditional mean, the blurry average over
all the plausible textures, because that is what minimizes mean absolute error. The 53.00 is the blur
tax, and it is largest precisely where capacity is tightest. This is not a learning-rate problem and not
a capacity problem I am allowed to touch; it is a *measurement* problem. The loss is optimizing the
wrong distance, and it is optimizing it hardest into the ground exactly where the model can least afford
to waste capacity on pixel-mean-matching.

So the obstacle is sharp: per-pixel L1 (and L2, and PSNR, which is L2 on a log axis) treats the image as
a bag of independent pixels, and that independence assumption is where it breaks. Blur a reconstruction
and the mean absolute residual can stay modest while the perceptual damage is obvious, because the
high-frequency structure that makes an image look sharp has been washed out and a pixel loss does not
know it should care. rFID, by contrast, compares the *distributions of deep features* between originals
and reconstructions — it is sensitive to exactly the softness a pixel loss is blind to. So I have an
objective (L1) that disagrees with my metric (rFID), and the small-scale 53.00 is what that disagreement
costs. The fix has to make the *training* loss see what rFID sees: I need to measure reconstruction
error somewhere that pixel softness is penalized, not in pixel space.

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

Now I have a choice about how much of that feature-space machinery to build myself. Raw deep-feature
distance is crude in a known way: convolutional channels have wildly different activation magnitudes, so
a naive sum of squared feature differences is dominated by a few high-energy channels and measures
whether those loud channels matched in *gain* rather than whether the *pattern* of activation agrees.
The mature fix is to unit-normalize across channels at each location (turning the comparison into a
gain-invariant cosine of feature direction), then recalibrate each channel by a learned non-negative
coefficient fit to human two-alternative-forced-choice judgments, spatially average within a layer, and
sum across layers. That whole object — a frozen trunk, channel normalization, learned per-channel
calibration grounded in human data — is precisely what the `lpips` package gives me, pre-built and
pre-calibrated. The edit surface hands me `lpips` as an available import, and the contract is that I
*consume* a perceptual distance, not that I learn one: I have no 2AFC judgments in this loop, no second
optimizer for calibration coefficients, no projection step. So the right move is not to re-derive and
re-fit the calibrated metric here — it is to take the off-the-shelf `lpips.LPIPS(net='vgg')`, freeze it
hard (eval mode, no gradients to its parameters), and use it as a fixed black-box perceptual ruler. The
emergent perceptual structure and the human calibration are baked in; my job is only to *measure with
it* and weight it against the terms I already have. Freezing it is not optional: if I let gradients into
the VGG trunk I would be retraining the very representation whose accidental perceptual goodness is the
thing I am exploiting, and I would bend it toward this one dataset; conservative read, calibrate
nothing, just measure. So `eval()`, `requires_grad_(False)` on every LPIPS parameter.

Now compose. I do not throw away the floor — the negative-ELBO skeleton was correct, the KL is still
the right closed-form leash at the same tiny `1e-6`, and I keep the L1 reconstruction term, because the
perceptual loss alone has a known weakness as a sole objective: optimizing it can leave high-frequency
artifacts, since many non-natural images map to similar deep features, so it wants a pixel anchor
alongside it for stability. The L1 term is that anchor; it pins coarse structure and color, while LPIPS
carries the perceptual sharpness the pixel term cannot see. So the reconstruction objective becomes
`L1(recon, target) + perceptual_weight · LPIPS(recon, target)`, and the full loss is that plus
`kl_weight · KL`. One implementation detail the harness forces: LPIPS is a VGG forward pass and the loop
runs under autocast mixed precision, so I cast both inputs to float before the perceptual call —
`self.lpips_fn(recon.float(), target.float())` — to keep the feature extractor in the precision it was
trained in and avoid half-precision drift in the distance; the `.mean()` reduces its per-image output to
a scalar.

The one real knob is `perceptual_weight`. The two terms are in different units — mean absolute pixel
error versus a calibrated perceptual distance — and I want the perceptual term to genuinely shape the
output without overpowering the pixel anchor that keeps coarse structure honest. A weight around `0.5`
is the standard balance for this pairing: large enough that LPIPS materially drives sharpness, small
enough that L1 still anchors. I keep `kl_weight = 1e-6` unchanged, because the floor's KL behavior was
not the problem — the reconstruction distance was — and there is no reason to disturb the leash. So the
literal edit is: in `__init__`, build and freeze `lpips.LPIPS(net='vgg').to(device)`, set
`perceptual_weight = 0.5` and `kl_weight = 1e-6`; in `forward`, compute L1, the float-cast mean LPIPS,
and the mean KL, sum them as `rec + 0.5·p_loss + 1e-6·kl`, and report all three in the metrics dict.
No discriminator, no frequency term yet — just the perceptual upgrade to the reconstruction distance.
(The distilled module is in the answer.)

Now the falsifiable expectations against the floor's numbers, because that is the point of having run
it. The whole diagnosis was that the blur tax is largest where capacity is tightest, and the perceptual
term attacks blur directly — so I expect the *biggest* improvement on the *small* scale, where L1 left
the most on the table at 53.00. That number should fall the hardest, plausibly by half or more, because
LPIPS refuses to reward the soft conditional mean the small model was hiding behind. The medium scale,
which started at 15.75 with moderate blur, should improve substantially too. The large scale was already
crisp at 5.35 — the pixel loss was nearly enough there — so the perceptual term has less blur to remove
and I expect a smaller absolute gain; it should still tick down, since deep-feature matching is strictly
more aligned with rFID than pixel matching, but I would be suspicious if it moved as much as the small
scale did. If instead the large scale improved most and the small barely moved, my whole "blur tax
scales inversely with capacity" reading would be wrong and I would have to rethink. PSNR and SSIM I
expect to hold or dip very slightly: those reward pixel and local-statistic agreement, which the pure L1
floor was already optimizing, and adding a perceptual term that tolerates small pixel-exact misses in
exchange for crispness can trade a hair of PSNR for a large rFID gain — which is the trade I want, since
rFID is the scored metric.

And I can already feel where this rung will run out. The perceptual loss pulls deep features together,
but it still does not have a sharp, adversarial opinion about whether a *patch* of texture is realistic —
VGG features can be matched by something that is close-but-still-a-bit-soft, because the metric is a
fixed distance, not an active critic hunting for the residual difference between a real image and a
reconstruction. So I expect perceptual to substantially close the small-scale gap but leave a residual
softness that a learned discriminator could attack — which is exactly the move the next rung makes,
adding an adversarial signal on top of this same L1 + LPIPS + KL skeleton.

The causal chain in one breath: L1's measured failure is a *measurement* problem — the small scale's
53.00 versus the large's 5.35 is the blur tax, largest where capacity is tightest, because a per-pixel
loss rewards the conditional-mean blur that rFID punishes → so measure reconstruction error in deep
feature space, where blur is penalized and distance is emergently perceptual → take the pre-calibrated
`lpips.LPIPS(net='vgg')` as a frozen black-box ruler rather than re-deriving its channel-normalization
and 2AFC calibration, because the harness exposes a perceptual distance to *consume*, not to *learn*,
and freezing it preserves the representation I am exploiting → keep the L1 term as a stability anchor
against perceptual high-frequency artifacts and keep the KL leash at `1e-6`, weighting LPIPS at `0.5` →
landing `F.l1_loss + 0.5·LPIPS + 1e-6·KL`, and expecting the small-scale 53.00 to fall the hardest with
diminishing gains up the capacity ladder, while a residual softness waits for an adversarial critic next.
