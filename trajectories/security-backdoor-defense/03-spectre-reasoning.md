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
— while a class has only a few thousand examples. I cannot robustly estimate a thousand-dimensional
covariance from that; the estimate would be garbage and whitening would inflate estimation noise rather
than poison. So I reduce dimension first: project the centered class features onto their top-`k`
singular subspace and do all robust estimation and whitening inside that `k`-dimensional space. This is
safe for the poison as long as `k` is large enough to contain the contaminated direction — the poison
bump, even if not the single top direction, is *some* above-average-variance direction, so it lives
within the top-`k` subspace for a reasonable `k` — and it makes robust estimation feasible (`k^2`
samples, not `d^2`) while restoring distance contrast.

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
