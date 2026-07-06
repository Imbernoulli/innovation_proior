The spectral rung did exactly what the derivation promised and then stopped exactly where I worried it
would, and the numbers draw the line cleanly. On resnet20-cifar10-badnets `poison_recall` climbed from
the clustering rung's 0.0016 to 0.2688, and on mobilenetv2-fmnist-badnets from 0.0000 to 0.1192 — the
spectral test is no longer blind; it surfaces real poison where 2-means found none, confirming that
reading the contaminated *direction* off the covariance beats asking a distance metric to find a compact
cluster in high dimensions. But look at `asr`: 0.9708 / 0.9999 / 1.0000, essentially unmoved from the
clustering rung. Recall went up, the backdoor did not break. And on vgg16bn-cifar100-blend recall stayed
flat at 0.0000 — the spectral test found nothing there at all. So `defense_score` (mean 0.4225) is, if
anything, a hair *below* the clustering rung's 0.4237, because the small recall gains did not translate
into any `(1-asr)` improvement and clean accuracy wobbled. The defense is detecting some poison and
still failing to defend.

Turn that 0.2688 into a body count, because it explains the flat `asr` exactly and sets the bar this
rung has to clear. On resnet20-cifar10-badnets there are about 2,500 poison in the target class, and the
harness removed the top `1.5*0.05*50,000 = 3,750` points. Recall 0.2688 means `0.2688 * 2,500 ≈ 672`
poison landed in the removed set; the other `1,828` — 73% of the poison — survived into the retrain set.
And 1,828 trigger-carrying, target-relabeled images is not a residue, it is a full-strength attack: it
is more poison than the entire cifar100 injection, and the retrain relearns the shortcut off it, which is
why `asr` reads 0.9708, essentially the clustering rung's value. The lesson in the arithmetic is that
`asr` is not linear in recall — it is a near-threshold: a backdoor trigger is a low-entropy, high-margin
feature that a network will relearn from even a small surviving fraction, so partial removal buys almost
nothing until removal is nearly total. The removal budget is not the obstacle — 3,750 comfortably exceeds
the 2,500 poison, so a ranking that floated *all* the poison to the top would remove every one and still
spend 1,250 slots on clean data. The obstacle is purely recall: to move `asr` I need recall not at 0.27
but near 1.0, which means catching the ~73% of poison the single top eigenvector never inspected. That is
the quantitative target — not "better detection" in the abstract, but the specific missing three-quarters
on cifar10 and the entire signature on cifar100.

The diagnosis is the limitation I named at the close of the last step, now measured. A 27% recall means
the top eigenvector caught roughly one poisoned point in four and missed three. The spectral test bets
everything on the *single* top-variance direction of the *combined* class covariance, and that bet only
pays off when the contaminated direction `Delta` is also the largest-variance direction of the data.
Two things break that bet here. First, the clean penultimate features of a class are not isotropic —
there are a few directions of large legitimate spread (pose, sub-type, lighting) and many of small
spread — so if the poison shift falls along a clean direction that is *not* the top one, the top
eigenvector is owned by clean variance and the poison projection looks clean. Second, the blended attack
on cifar100 spreads its trigger more diffusely than a single bright BadNets pixel, so its signature is
not one sharp direction but several low-variance ones; the top eigenvector inspects exactly one of them,
which is why recall there is pinned at zero. The 0.27 on cifar10 is the fraction of poison that happened
to align with the one direction I looked at; the missing 0.73 is sitting in directions the top
eigenvector never inspected. So the failure is structural: a single principal direction cannot catch a
signature that is not the loudest direction, and an adversary (or just an ill-conditioned clean
covariance) can put it anywhere. I do not need a *better* top eigenvector — I need the poison's excess
variance to stand out *wherever it sits in the spectrum*.

The geometric operation that makes excess variance stand out regardless of direction is **whitening**.
Suppose I knew the *clean* covariance `Sigma_clean` of the class. Whiten every point by
`Sigma_clean^{-1/2}`: the clean cloud becomes isotropic, unit variance in every direction. After that
transform there is no longer a "big clean direction" for the poison to hide behind — every clean
direction is equalized. The poison's mean shift `Delta`, a fixed vector in feature space, becomes
`Sigma_clean^{-1/2} Delta` in whitened space, and its excess variance is now the *only* above-unit-
variance structure left, exposed no matter which clean direction it originally overlapped. Whitening
converts the question I cannot control — "is the signature a top-variance direction?" — into one whose
answer is yes whenever poison is present — "is there *any* residual high-variance direction?" That is
exactly the lever the single-eigenvector test was missing, and it directly attacks the cifar100 zero and
the cifar10 three-quarters-missed.

But this trades the problem for a harder one: I do not have `Sigma_clean`. All I have is the
*contaminated* class covariance, and if I whiten by *that*, I whiten away the poison along with
everything else — the contaminated covariance already contains the rank-one bump `eps(1-eps)Delta
Delta^T`, so `Sigma_F^{-1/2}` flattens precisely the direction I want to keep tall. The whole approach
hinges on estimating the *clean* covariance from contaminated samples, with no trusted clean set. This
is the exact problem high-dimensional robust statistics solves: estimate the mean and covariance of `D`
when an `eps`-fraction of samples are adversarial outliers. Robust estimators recover the clean mean and
covariance up to error `O(eps sqrt(log(1/eps)))` from enough samples — close enough that whitening by
the robust covariance makes the clean cloud nearly isotropic while leaving the poison's excess variance
intact. So robust whitening is the amplifier: estimate the clean covariance robustly, whiten by it, and
*then* the residual signature pops.

How to do the robust estimation operationally. The full polynomial-time estimators are intricate; the
workhorse underneath them is iterative filtering. Start with all points, compute a mean and covariance,
measure each point's Mahalanobis distance under the current estimate, keep the points closest to the
running core and drop the most extreme, recompute, repeat. The fixed point is a clean core whose mean
and covariance are not dominated by the outliers, because the outliers — the high-Mahalanobis-distance
points — are exactly the ones trimmed. The trim fraction should track the contamination level (on the
order of `eps`, with a margin, since over-trimming clean points costs almost nothing while under-
trimming leaves poison in the covariance and re-pollutes the whitening — the same asymmetry the harness's
`1.5*eps` removal already encodes). A handful of iterations converges; because the trim fraction is
below a half, the clean majority is never trimmed away.

There is a dimensionality obstacle I have to clear before whitening, and it is the difference between
this working and amplifying noise. Robust covariance estimation in `d` dimensions needs on the order of
`d^2` samples to be accurate, and `d` here is the penultimate-feature dimension — hundreds to thousands
— while a class has only a few thousand examples. Put the numbers in: MobileNetV2's penultimate width is
`d = 1,280`, so a full covariance has `d(d+1)/2 ≈ 820,000` free parameters, and `d^2 ≈ 1.64` million —
against a FashionMNIST class of ~6,000 points. VGG-16-BN is `d = 512`, `d^2 ≈ 262,000`, against a
cifar100 class of ~500-1,000 points, worse by two orders of magnitude. The sample-to-parameter ratio is
far below one; the covariance estimate would be rank-deficient and its inverse-square-root — the
whitening map — would blow up the many near-zero sample directions, inflating estimation noise rather
than poison. I cannot robustly estimate a thousand-dimensional covariance from a few thousand points. So I reduce dimension first: project the centered class features onto their top-`k`
singular subspace and do all robust estimation and whitening inside that `k`-dimensional space. This is
safe for the poison as long as `k` is large enough to contain the contaminated direction — the poison
bump, even if not the single top direction, is *some* above-average-variance direction, so it lives
within the top-`k` subspace for a reasonable `k` — and it makes robust estimation feasible (`k^2`
samples, not `d^2`) while restoring distance contrast. With the cap at `k = 64`, `k^2 = 4,096`, which sits
below the cifar10 and fmnist class sizes (thousands) and near the cifar100 class size — feasible where the
full `d^2 = 262,000` was not, a two-orders-of-magnitude reduction in what the robust estimator has to
support. It also flips the distance-concentration argument that killed the clustering rung: at `k=64` the
relative distance spread `~1/sqrt(k) ≈ 0.125` is back to where contrast exists, so the Mahalanobis
trimming the robust estimator relies on can actually separate core from outlier.

But `k` is a real knob with failure on both sides, and the spectral rung's cifar100 zero is a warning
that fixed choices can be exactly wrong. Too small a `k`: the subspace might not contain the poison
direction, and I project the signature away before I start. Too large a `k`: I drag in clean directions
that are heavy-tailed or poorly estimated from limited samples, the robust estimator mis-whitens them,
and whitening *inflates* a clean direction so the score flags clean points. The sweet spot is attack-
dependent — a sharp BadNets trigger has low effective dimension and wants a small `k`; a diffuse blend or
a spread signature wants a larger `k`. I refuse to be on a knife's edge about a number that is wrong for
half the settings, especially after watching one fixed top-eigenvector choice fail on cifar100. So I
*select* `k` by the signal it produces: run the pipeline for a grid of candidate `k`, and for each
measure how strong the residual signature is *after* whitening — the top eigenvalue of the whitened
empirical covariance. A `k` that contains the poison and estimates the clean directions well leaves a
tall residual eigenvalue (the un-whitened poison); a `k` too small (no poison) or too large (mis-whitened
clean dilution) leaves a flatter whitened spectrum. Pick the `k` that maximizes the post-whitening top
eigenvalue. The data tells me the effective dimensionality of the signature instead of my guessing it;
a geometric grid from 1 up to a cap suffices and the choice is not sensitive to the exact grid.

Trace why that argmax points the right way in each regime, because a selection rule is only as good as
its extremum. If `k` is *too small* to contain the poison direction, the top-`k` subspace is all clean
variance; the robust estimator whitens it to near-isotropy and the post-whitening top eigenvalue sits
near 1 — a flat signal, correctly rejected. If `k` is *right*, the subspace contains the contaminated
direction, but the robust estimator trims the poison as high-Mahalanobis outliers and so does not count
it toward the clean covariance; whitening therefore leaves that direction un-shrunk and the
post-whitening top eigenvalue reads well above 1 — the tall signal I select. If `k` is *too large*, I drag
in clean directions estimated from too few samples relative to `k^2`, the robust estimator mis-whitens
them, and one may inflate above 1 — a false tall signal, and I have to be honest the argmax could chase
it. Two things bound the damage: the `k = 64` cap keeps `k^2` under the class sizes so mis-whitening
stays mild, and a mis-whitened clean direction inflates *diffusely* across many points rather than
isolating a minority, so it does not manufacture the sharp poison/clean split a true signature does — the
score degrades gracefully rather than flipping to flag clean data. The rule is not perfect, but its
failure mode is bounded and its success mode is exactly the "un-whitened poison direction stays tall"
event I want, so I trust the argmax over a capped geometric grid more than any fixed `k` — which is what
the spectral rung effectively committed to and what its cifar100 zero punished. I choose iterative
Mahalanobis trimming for the robust step over the alternatives on the same feasibility grounds: a
minimum-covariance-determinant search is combinatorial and far too slow to run per class per `k`, and a
fixed per-coordinate quantile clip ignores exactly the joint covariance structure whitening exists to
exploit, whereas trimming by Mahalanobis distance under the running estimate is the cheapest operation
that removes the joint outliers the whitening must not see.

After robust whitening, how do I score? The two endpoints I already know are both wrong for half the
cases. Score by the squared whitened *norm*: the total excess across all directions, good when the
signature is *spread* (an m-way or diffuse blend), bad when it is one sharp direction (diluted by `k-1`
unit-variance clean directions in the sum). Score by the squared projection onto the whitened *top
eigenvector*: perfect for one sharp direction (this is what the spectral rung did, post-whitening), but
throws away the signal when the signature is spread — which is precisely the cifar100 case that scored
zero. I have spent this whole rung making the method handle the spread case, so I cannot commit to the
top-projection; but I also do not want to lose the sharp cifar10 case to the norm score. I want a score
that *interpolates* and adapts to the effective dimensionality automatically.

That is exactly the quantum-entropy score. Take the empirical covariance `Sigma_tilde` of the whitened
reps, form `Q = exp(alpha (Sigma_tilde - I)/(||Sigma_tilde||_2 - 1))`, normalize by `Tr(Q)`, and score
each whitened point by `tau_i = (h_i^T Q h_i)/Tr(Q)`. Read what `Q` does. After whitening the clean
directions have variance near 1, so `Sigma_tilde - I` is near zero on them and `Q` weights them lightly;
the contaminated directions have variance above 1, so `Sigma_tilde - I` is positive there and the
exponential weights them *heavily*. The normalization `||Sigma_tilde||_2 - 1` scales the temperature by
the strength of the top excess direction so it is comparable across classes and settings. The single
parameter `alpha` controls how aggressively `Q` concentrates: as `alpha -> 0`, `Q -> I` and the score is
the squared whitened norm (the spread-friendly endpoint); as `alpha -> infinity`, `Q` collapses onto the
top whitened eigenvector and the score is the top-projection (the sharp endpoint); intermediate `alpha`
weights each excess direction by `exp(alpha * its excess)`, a soft-max over directions that automatically
follows whatever effective dimensionality the whitening exposed. A sharp BadNets trigger leaves one tall
direction and `Q` concentrates there; a diffuse blend leaves several moderately-tall directions and `Q`
spreads weight across them. `alpha = 4` sits in the regime that handles both — large enough to emphasize
excess over clean noise, small enough not to bet everything on the single top direction the spectral rung
already over-bet on. The QUE score is not an arbitrary nonlinearity; it is the precise object that makes
the method agnostic to how concentrated or spread the signature is, which is the entire reason I did
robust whitening.

I should verify the two endpoint claims literally rather than assert them, since the whole argument for
QUE rests on it interpolating correctly. Take `alpha = 0`: then `A = 0`, `Q = exp(0) = I`, and the score
is `h_i^T I h_i / Tr(I) = ||h_i||^2 / k` — exactly the squared whitened norm, the spread-friendly
endpoint. Take `alpha -> infinity`: `A` is a positive scaling of `Sigma_tilde - I`, whose top eigenvector
`e_1` carries the largest excess; `exp(alpha A)` has eigenvalues `exp(alpha * excess_j)`, so the largest
dominates and `Q / Tr(Q) -> e_1 e_1^T`, giving score `-> (h_i . e_1)^2`, the squared top-projection — the
sharp endpoint, which is precisely the spectral rung's test applied in whitened space. So `alpha` really
does slide between the two failures I already know. Now put `alpha = 4` on a concrete whitened spectrum
to see the emphasis. Suppose after whitening a class's `Sigma_tilde` has one excess direction at variance
3 and the rest at 1, so in that direction `Sigma_tilde - I = 2` and the top norm is `||Sigma_tilde||_2 =
3`, `denom = max(3-1, 1e-6) = 2`. Then `A` on that direction is `4 * 2 / 2 = 4` and on clean directions
`0`, so `Q` weights the excess direction by `exp(4) ≈ 54.6` and each clean direction by `exp(0) = 1`. A
poison point aligned with the excess is scored ~55x more heavily than the clean spread it used to be
diluted by — that is the amplification the raw squared-projection sum in the spectral rung could not
supply. Now stress the *spread* case: let `Sigma_tilde = diag(2, 2)` in a 2-D whitened subspace (two
equally-excess directions). Then `||Sigma_tilde||_2 = 2`, `denom = 1`, `A = 4*(diag(2,2) - I)/1 =
diag(4,4)`, so `Q = diag(e^4, e^4)` weights both directions equally and the score collapses back to
`(h_1^2 + h_2^2) * e^4 / (2 e^4) = ||h||^2 / 2` — the norm. So on a genuinely spread signature QUE
*automatically* becomes the spread-friendly norm, and on a sharp one it concentrates 55-fold; the single
`alpha = 4` did both without my telling it which case it was in. That is the check that convinces me QUE
is the right score and not just a plausible-looking exponential.

Ground it in the scaffold. Same `BackdoorDefense` contract, same per-training-label routing as the
spectral rung (the target label is not exposed, so run on every class; group by cached training labels,
not predicted class, for the same reason as before — a hard poisoned point's prediction can disagree with
its label). Per class: center, then for each candidate `k` project onto the top-`k` subspace, robustly
estimate the clean mean/covariance by iterative Mahalanobis trimming, whiten by the robust
`Sigma^{-1/2}`, record the post-whitening top eigenvalue, keep the best `k`, and compute the QUE score
for it. Numerical guards fall out of the construction: clip the matrix-exponential exponent so it cannot
overflow, add a small ridge so the robust covariance is invertible, skip classes too small to whiten,
and fall back to the squared whitened norm if the trace degenerates. `fit` does the per-class work and
caches the scores; `score_samples` reads them out. The harness's fixed `1.5*eps` removal does the rest.

Here is the bar this rung must clear, falsifiably, against the spectral rung's real numbers, since it
carries no feedback of its own. The spectral rung left recall at 0.2688 / 0.0000 / 0.1192 with `asr` at
0.9708 / 0.9999 / 1.0000 and `defense_score` mean 0.4225. Robust whitening should raise recall on
cifar10 well above 0.2688, because the poison the top eigenvector missed (the other three-quarters) is
exactly the excess variance whitening exposes and QUE then weights; and it should move cifar100 off zero,
because the diffuse blend signature that one direction could not see is what whitening-plus-QUE is built
to surface. The real test is `asr`: recall only matters if removing the top `1.5*eps` now catches *enough*
of the poison to break the trigger shortcut — if recall crosses the threshold where the retrained model
no longer relearns the backdoor, `asr` should finally fall from ~1.0 and `defense_score` should jump well
past 0.4225 toward the high-clean-acc, low-asr corner. If instead recall rises but `asr` stays pinned, the
verdict is that within-class poison fraction or trigger diffuseness has pushed the setting outside the
regime where *any* feature-scoring filter at a `1.5*eps` budget can break the backdoor — a limit of the
filter-and-retrain protocol itself, not of the scoring rule. That is the line I would be testing, and it
is the honest place this ladder ends. (The full scaffold module is in the answer.)
