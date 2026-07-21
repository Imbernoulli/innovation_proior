The spectral test did what the derivation promised and then stopped exactly where I worried, and the
numbers draw the line. On resnet20-cifar10-badnets `poison_recall` climbed from clustering's
0.0016 to 0.2688, and on mobilenetv2-fmnist-badnets from 0.0000 to 0.1192 — the spectral test surfaces
real poison where 2-means found none, confirming that reading the contaminated *direction* off the
covariance beats asking a distance metric for a compact cluster. But `asr` is 0.9708 / 0.9999 / 1.0000,
unmoved from clustering. Recall went up, the backdoor did not break. And cifar100 recall stayed flat at
0.0000 — the spectral test found nothing there. So `defense_score` (mean 0.4225) is if anything a hair
below clustering's 0.4237: the small recall gains did not translate into any `(1-asr)` improvement.

Turn 0.2688 into a body count, because it explains the flat `asr` and sets the bar. On cifar10 there are
~2,500 poison in the target class and the harness removed the top `1.5*0.05*50,000 = 3,750`. Recall
0.2688 means ~672 poison landed in the removed set; the other ~1,828 — 73% of the poison — survived. And
1,828 trigger-carrying, target-relabeled images is not a residue, it is a full-strength attack — more
poison than the entire cifar100 injection — and the retrain relearns the shortcut off it, which is why
`asr` reads essentially the clustering value. `asr` is not linear in recall; it is a near-threshold,
because a backdoor trigger is a low-entropy, high-margin feature the network relearns from even a small
surviving fraction, so partial removal buys almost nothing until removal is nearly total. The budget is
not the obstacle — 3,750 comfortably exceeds 2,500. The obstacle is purely recall: to move `asr` I need
recall near 1.0, catching the ~73% the single top eigenvector never inspected.

The diagnosis is the limitation I named last step, now measured. The spectral test bets everything on the
*single* top-variance direction of the *combined* class covariance, and that pays off only when `Delta`
is also the largest-variance direction. Two things break it here. First, the clean features are not
isotropic — a few directions of large legitimate spread (pose, sub-type, lighting) and many of small — so
if the poison shift falls along a clean direction that is not the top one, the top eigenvector is owned by
clean variance and the poison projection looks clean. Second, the blended attack on cifar100 spreads its
trigger more diffusely than a single bright BadNets pixel, so its signature is several low-variance
directions and the top eigenvector inspects one of them — which is why recall there is pinned at zero. So
the failure is structural: a single principal direction cannot catch a signature that is not the loudest
direction. I do not need a *better* top eigenvector — I need the poison's excess variance to stand out
*wherever* it sits in the spectrum.

The operation that makes excess variance stand out regardless of direction is **whitening**. If I knew
the *clean* covariance `Sigma_clean` of the class, whiten every point by `Sigma_clean^{-1/2}`: the clean
cloud becomes isotropic, unit variance in every direction, so there is no longer a "big clean direction"
for the poison to hide behind. The poison's mean shift `Delta` becomes `Sigma_clean^{-1/2} Delta`, and
its excess variance is now the *only* above-unit-variance structure left, exposed no matter which clean
direction it originally overlapped. Whitening converts the question I cannot control — "is the signature a
top-variance direction?" — into one whose answer is yes whenever poison is present — "is there *any*
residual high-variance direction?"

But I do not have `Sigma_clean`. All I have is the *contaminated* class covariance, and whitening by
*that* whitens away the poison with everything else — the contaminated covariance already contains the
rank-one bump `eps(1-eps)Delta Delta^T`, so `Sigma_F^{-1/2}` flattens precisely the direction I want to
keep tall. The whole approach hinges on estimating the *clean* covariance from contaminated samples with
no trusted clean set. This is exactly what high-dimensional robust statistics solves: estimate the mean
and covariance of `D` when an `eps`-fraction of samples are adversarial outliers, up to error
`O(eps sqrt(log(1/eps)))`. Whitening by the robust covariance makes the clean cloud nearly isotropic while
leaving the poison's excess variance intact.

Operationally, the workhorse under the polynomial-time robust estimators is iterative filtering. Start
with all points, compute a mean and covariance, measure each point's Mahalanobis distance under the
current estimate, keep the closest to the running core and drop the most extreme, recompute, repeat. The
fixed point is a clean core whose moments are not dominated by the outliers, because the high-Mahalanobis
points — the poison — are trimmed. The trim fraction should track `eps` with a margin: over-trimming
clean points costs almost nothing while under-trimming leaves poison in the covariance and re-pollutes
the whitening — the same asymmetry the harness's `1.5*eps` removal encodes. A handful of iterations
converges, and because the trim is below a half the clean majority is never trimmed away.

There is a dimensionality obstacle before whitening, and it is the difference between this working and
amplifying noise. Robust covariance estimation in `d` dimensions needs on the order of `d^2` samples, and
`d` here is hundreds to thousands while a class has only a few thousand examples. MobileNetV2's `d =
1,280` gives `d(d+1)/2 ≈ 820,000` free parameters and `d^2 ≈ 1.64` million against a FashionMNIST class
of ~6,000; VGG-16-BN's `d = 512` gives `d^2 ≈ 262,000` against a cifar100 class of ~500-1,000, worse by
two orders of magnitude. The estimate would be rank-deficient and its inverse-square-root — the whitening
map — would blow up the many near-zero directions, inflating noise rather than poison. So I reduce
dimension first: project the centered class features onto their top-`k` singular subspace and do all
robust estimation and whitening there. Safe for the poison as long as `k` contains the contaminated
direction — the poison bump, even if not the single top direction, is *some* above-average-variance
direction, so it lives within the top-`k` subspace for reasonable `k` — and it makes robust estimation
feasible (`k^2` samples, not `d^2`) while restoring distance contrast. Capped at `k = 64`, `k^2 = 4,096`
sits below the cifar10 and fmnist class sizes and near the cifar100 size, and the relative distance spread
`~1/sqrt(64) ≈ 0.125` is back where the Mahalanobis trimming has contrast to separate core from outlier.

But `k` is a real knob with failure on both sides, and the spectral test's cifar100 zero is a warning
that fixed choices can be exactly wrong. Too small a `k`: the subspace might not contain the poison
direction, and I project the signature away before I start. Too large a `k`: I drag in clean directions
that are heavy-tailed or poorly estimated, the robust estimator mis-whitens them, and whitening
*inflates* a clean direction so the score flags clean points. The sweet spot is attack-dependent — a
sharp BadNets trigger has low effective dimension and wants a small `k`; a diffuse blend wants a larger
`k`. Rather than sit on a knife's edge, I *select* `k` by the signal it produces: run the pipeline for a
grid of candidate `k` and, for each, measure how strong the residual signature is *after* whitening — the
top eigenvalue of the whitened empirical covariance. A `k` that contains the poison and estimates the
clean directions well leaves a tall residual eigenvalue (the un-whitened poison); a `k` too small (no
poison) or too large (mis-whitened clean dilution) leaves a flatter spectrum. Pick the `k` maximizing the
post-whitening top eigenvalue — the data reveals the signature's effective dimensionality instead of my
guessing it. A geometric grid from 1 to the cap suffices.

The argmax points the right way in each regime, and its failure mode is bounded. Too small, the subspace
is all clean variance, whitened to near-isotropy, post-whitening top eigenvalue near 1 — correctly
rejected. Right, the subspace contains the contaminated direction, the robust estimator trims the poison
as outliers so does not count it toward the clean covariance, whitening leaves that direction un-shrunk
and the top eigenvalue reads well above 1 — the tall signal I select. Too large, mis-whitened clean
directions can inflate above 1, a false tall signal the argmax could chase — but the `k = 64` cap keeps
`k^2` under the class sizes so mis-whitening stays mild, and a mis-whitened clean direction inflates
*diffusely* across many points rather than isolating a minority, so it degrades the score gracefully
rather than flipping to flag clean data. For the robust step itself I choose iterative Mahalanobis
trimming over the alternatives on feasibility: a minimum-covariance-determinant search is combinatorial
and far too slow to run per class per `k`, and a per-coordinate quantile clip ignores the joint
covariance structure whitening exists to exploit.

After robust whitening, how to score? The two endpoints I know are each wrong for half the cases. Squared
whitened *norm* — the total excess across all directions — is good when the signature is *spread* (a
diffuse blend), bad when it is one sharp direction diluted by `k-1` unit-variance clean directions.
Squared projection onto the whitened *top eigenvector* — the spectral test post-whitening — is
perfect for one sharp direction but throws away a spread signature, precisely the cifar100 case that
scored zero. I have spent this step making the method handle the spread case, so I cannot commit to
top-projection, but I do not want to lose the sharp cifar10 case to the norm. I want a score that
interpolates and adapts to the effective dimensionality automatically.

That is the quantum-entropy score. Take the empirical covariance `Sigma_tilde` of the whitened reps, form
`Q = exp(alpha (Sigma_tilde - I)/(||Sigma_tilde||_2 - 1))`, normalize by `Tr(Q)`, and score each whitened
point by `tau_i = (h_i^T Q h_i)/Tr(Q)`. After whitening the clean directions have variance near 1, so
`Sigma_tilde - I` is near zero there and `Q` weights them lightly; the contaminated directions have
variance above 1, so `Q` weights them *heavily*. The normalization `||Sigma_tilde||_2 - 1` scales the
temperature by the top excess direction's strength so it is comparable across classes. `alpha` controls
concentration: as `alpha -> 0`, `Q -> I` and the score is the squared whitened norm (the spread
endpoint); as `alpha -> infinity`, `Q` collapses onto the top whitened eigenvector and the score is the
top-projection (the sharp endpoint); intermediate `alpha` weights each excess direction by
`exp(alpha * its excess)`, a soft-max over directions that follows whatever effective dimensionality the
whitening exposed. `alpha = 4` sits in the regime that handles both — large enough to emphasize excess
over clean noise, small enough not to bet everything on the single top direction the spectral test
over-bet on. On a whitened spectrum with one excess direction at variance 3 and the rest at 1, `denom =
max(3-1, 1e-6) = 2`, so `A` on that direction is `4*2/2 = 4` and `Q` weights it by `exp(4) ≈ 55x` the
clean directions — the amplification the raw squared-projection sum could not supply — while on a
genuinely spread spectrum (several equally-excess directions) the exponentials equalize and the score
collapses back to the norm. One `alpha` does both without my telling it which case it is in.

Ground it in the scaffold. Same `BackdoorDefense` contract, same per-training-label routing as the
spectral test (target label not exposed, so run every class; group by cached training labels, not
predicted class, so a hard poisoned point whose prediction disagrees is not misrouted). Per class:
center, then for each candidate `k` project onto the top-`k` subspace, robustly estimate the clean
mean/covariance by iterative Mahalanobis trimming, whiten by the robust `Sigma^{-1/2}`, record the
post-whitening top eigenvalue, keep the best `k`, and compute the QUE score for it. Numerical guards fall
out of the construction: clip the matrix-exponential exponent against overflow, add a small ridge so the
robust covariance is invertible, skip classes too small to whiten, and fall back to the squared whitened
norm if the trace degenerates. `fit` does the per-class work and caches the scores; `score_samples` reads
them out.

The bar to clear, measured against the previous step's numbers: robust whitening should raise cifar10
recall well above 0.2688 — the poison the top eigenvector missed is exactly the excess variance whitening
exposes and QUE weights — and move cifar100 off zero, because the diffuse blend one direction could not
see is what whitening-plus-QUE is built to surface. The real test is `asr`: recall matters only if
removing the top `1.5*eps` now catches *enough* to break the trigger shortcut. If recall crosses the
near-threshold where the retrain no longer relearns the backdoor, `asr` should fall from ~1.0 and
`defense_score` jump well past 0.4225 toward the high-clean-acc, low-asr corner. If recall rises but `asr`
stays pinned, the verdict is that within-class poison fraction or trigger diffuseness has pushed the
setting outside the regime where *any* feature-scoring filter at a `1.5*eps` budget can break the
backdoor — a limit of the filter-and-retrain protocol itself, not of the scoring rule. That is the honest
place this line of attack ends. The full module is in the answer.
