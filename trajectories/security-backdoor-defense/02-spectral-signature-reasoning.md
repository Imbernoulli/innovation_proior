The clustering rung came back almost exactly as I feared, and the numbers say *why* in detail. On
resnet20-cifar10-badnets `poison_recall` is 0.0016 — out of the `1.5*eps` points the harness removed, it
caught essentially none of the real poison; on vgg16bn-cifar100-blend and mobilenetv2-fmnist-badnets the
recall is a flat 0.0000. And the consequence shows up exactly where I said it would: with the trigger
shortcut left intact, `asr` stays pinned at 0.97 / 1.00 / 1.00, and `defense_score` (mean 0.4237) is
just clean accuracy carried by the retrain plus a `(1-asr)` term that is near zero. The defense did
nothing; the retrained model relearned the backdoor from the poison the filter failed to remove.

Read the score formula against these numbers, because it tells me where the points actually come from
and where the ceiling is. `defense_score = 0.5*clean_acc + 0.5*(1-asr)`. On resnet20-cifar10-badnets that
is `0.5*0.9209 + 0.5*(1-0.9729) = 0.4605 + 0.0136 = 0.4740` — the reported number to the digit, and the
arithmetic makes plain that the `(1-asr)` half contributed a rounding crumb (0.0136 out of 0.5) while
clean accuracy carried the whole score. The three settings differ almost entirely through clean
accuracy: cifar10 and fmnist score ~0.474 because their retrained clean accuracy is ~0.92-0.95 with
`asr` pinned, while cifar100 scores only 0.3234 because a 100-way problem retrains to 0.6467 clean
accuracy — `0.5*0.6467 + 0.5*(1-1.0000) = 0.3234`, again exact. So the objective is currently a clean-
accuracy readout with a dead defense term, and the only way to move it is to make `asr` actually fall,
which requires removing enough poison that the retrain forgets the trigger. The clean-accuracy ceiling
per setting (if I drove `asr` to 0 without hurting clean acc) would be `0.5*clean_acc + 0.5`: about 0.96
on cifar10, 0.82 on cifar100, 0.97 on fmnist. That is the prize, and every point of it is gated behind
detection I do not yet have. This is
the high-dimensional-clustering collapse I flagged in the last step, confirmed: 2-means run directly on
the penultimate features did not find a clean-versus-poison split, it bisected one blob arbitrarily, so
the "smaller cluster" was half of the clean data and the score ranked clean points as suspicious. The
diagnosis is sharp and it is not a tuning problem. Distance-based clustering needs distance contrast,
and in hundreds of feature dimensions the distances concentrate, so the contrast the method depends on
is gone. The lesson the recall numbers teach me: I should not be asking k-means to *partition* the cloud
into two compact blobs at all. I should find the single *direction* the poison shift inflates and
measure along it — a spectral question, not a clustering one — because a direction can be read off the
covariance even when no compact cluster is visible to a distance metric.

Let me set up that direction precisely, because "find the poison direction" has to be derived, not
gestured at. Within one class the training points are a mixture `F = (1-eps) D + eps W` of clean points
`D` (mean `mu_D`, fraction `1-eps`) and poison `W` (mean `mu_W`, fraction `eps`), with `Delta = mu_D -
mu_W` the gap between the two sub-population means. If `D` and `W` differ enough in mean, there is a
direction along which I can threshold to separate them; the question is whether the *covariance* tells
me that direction. Compute the mixture covariance about the mixture mean `mu_F = (1-eps)mu_D +
eps mu_W`. For the clean part, `mu_D - mu_F = eps Delta`, so `E_{X~D}[(X-mu_F)(X-mu_F)^T] = Sigma_D +
eps^2 Delta Delta^T`; for the poison part, `mu_W - mu_F = -(1-eps)Delta`, so `E_{X~W}[...] = Sigma_W +
(1-eps)^2 Delta Delta^T`. Mixing with weights `(1-eps)` and `eps`, the `Delta Delta^T` coefficient is
`(1-eps)eps^2 + eps(1-eps)^2 = eps(1-eps)`, giving

  `Sigma_F = (1-eps) Sigma_D + eps Sigma_W + eps(1-eps) Delta Delta^T`.

There it is. The contamination contributes a *rank-one* variance bump `eps(1-eps) Delta Delta^T` on top
of the within-population covariances. The mean shift between clean and poison announces itself as extra
variance pointing exactly along `Delta`. So if that bump is large relative to the within-population
spread, the top eigenvector of `Sigma_F` lines up with `Delta`, and the squared projection onto it
separates poison from clean. This is the lever clustering missed: I do not need to *group* the points,
I need the one direction of anomalous variance, and the covariance hands it to me directly. Crucially,
reading a covariance eigenvector does not suffer the distance-concentration that killed 2-means — the
spectral signal is in the second moment, not in pairwise distances.

Why would this succeed where pixel-level analysis fails — and where, frankly, the clustering also had to
work? Because the size of the bump `eps(1-eps)||Delta||^2` has to beat the within-population variance to
dominate the spectrum. At the pixel level `||Delta||` is tiny (a trigger barely moves the mean) while
the natural-image variance is enormous, so the bump is buried. In the *learned representation* the
network amplifies the trigger — it is rewarded for keying on a near-perfect predictor of the target — so
`||Delta||` in penultimate-feature space is large, pushed past the within-class variance. The same
amplification I leaned on for clustering is what makes the spectral bump visible; the difference is that
the spectral test reads the bump off the covariance rather than asking a distance metric to find a
cluster, so it survives the high dimension that defeated 2-means. This is exactly why I expect this rung
to move recall off the floor where clustering left it.

I owe myself the condition under which this is guaranteed, because "should separate" is what I said
about clustering too. Call `D, W` *eps-spectrally separable* if for the top eigenvector `v` of `Sigma_F`
there is a threshold `t>0` with `Pr_{D}[|<X-mu_F,v>| > t] < eps` (almost no clean point projects beyond
`t`) and `Pr_{W}[|<X-mu_F,v>| < t] < eps` (almost no poison point within it). Then removing the largest
projections removes nearly all poison while sacrificing little clean data. The claim: if `Sigma_D,
Sigma_W <= sigma^2 I` and `||Delta||^2 >= 6 sigma^2/eps` (with `eps < 1/2`), then `D, W` are
eps-spectrally separable. Three moves. (1) Chebyshev: along any unit `u`, each sub-population
concentrates within `~sigma` of its own mean, `Pr_D[|<X-mu_D,u>| > t] <= sigma^2/t^2`. (2) Correlation
implies separation: with `c = |<u,Delta>|`, the clean and poison projected centers about `mu_F` sit at
`eps c` and `-(1-eps)c`; a threshold between them with Chebyshev room on both sides drives both error
tails below `eps` once `c` is large enough relative to `sigma/sqrt(eps)`. (3) The top eigenvector is
correlated with `Delta`: from `Sigma_F >= eps(1-eps) Delta Delta^T`, the top eigenvalue is at least
`eps(1-eps)||Delta||^2`, and expanding `v^T Sigma_F v <= sigma^2 + eps(1-eps)<v,Delta>^2` gives
`<v,Delta>^2 >= ||Delta||^2 - sigma^2/(eps(1-eps))`; under the `6`-constant hypothesis and `eps<1/2`
this is `> (2/3)||Delta||^2 > 4 sigma^2/eps`, supplying the correlation move (2) needs. The constant `6`
is a proof-sketch constant, not an optimized threshold; the finite-sample version costs a little slack
(`eps < 1/4`, `||Delta||^2 >= 10 sigma^2/eps`, `n = Omega(d log n/eps)` samples) via a
matrix-concentration bound, and gives separation with probability at least 9/10.

Put a number to that threshold so I know what it demands, using the within-class poison fractions I can
work out from the setup. On resnet20-cifar10-badnets the target class holds ~2,500 poison among ~5,000
clean, a within-class `eps ≈ 1/3`, so the condition `||Delta||^2 >= 6 sigma^2/eps` reads
`||Delta|| >= sqrt(6/0.33) sigma ≈ 4.2 sigma` — the poison mean must sit more than four within-population
standard deviations off the clean mean *along the recovered direction* for guaranteed separation. That
is a stiff bar, but it is exactly the bar the trigger amplification is built to clear: the network is
rewarded for making the trigger a near-perfect, low-variance predictor, which is precisely a large-mean-
shift, small-`sigma` sub-population. Where amplification is only partial, or where the sample count `n`
per class is small next to `d` (CIFAR-100 classes have ~500-1,000 points against `d=512`, badly violating
`n = Omega(d log n / eps)`), the finite-sample guarantee lapses and I should expect a recovered direction
polluted by estimation noise. So the arithmetic predicts a graded outcome: cleanest on the low-`d`,
high-`n` cifar10 badnets setting, shakiest on cifar100 where the sample budget is thin and the within-
class fraction is nearest the `1/2` cliff.

This derivation also tells me the regime where the method *breaks*, and it is exactly the regime the
task-design notes warned about. The whole argument needs `eps < 1/2` within the class. If the within-
class poison fraction climbs past a half, the "outlier" sub-population becomes the majority, centering
subtracts a mean dominated by poison, and the *clean* images become the extreme points of `v` — I would
remove clean data. That is why the vgg16bn-cifar100-blend setting uses 1% global poison (keeping the
target class ~33% poison) rather than 5% (which would make the target class 83% poison): at 83% the
spectral test is pointed at the wrong sub-population by construction. So on cifar100 I should not expect
miracles even from a faithful spectral test — but I should at least expect it to stop being blind, which
is the bar the clustering rung failed.

Before I commit to the single top eigenvector, I owe the alternative a fair walk, because "take the top
direction" is one choice among several and the wrong one is tempting. The obvious upgrade is to score by
the summed squared projection onto the top-`m` eigenvectors, `sum_{j<=m} ((r_i - mu_c) . v_j)^2`, on the
theory that if the poison shift spreads across a few directions I would catch more of it. Walk it a few
steps and it falls apart on its own terms. The rank-one bump `eps(1-eps) Delta Delta^T` inflates exactly
one direction; the *other* top eigenvectors `v_2, ..., v_m` are, by construction, the directions of
largest *clean* variance (pose, sub-type, lighting), each contributing a squared projection of order
`sigma_clean^2` that is the same for poison and clean points. So summing them in adds `m-1` terms of
pure clean spread to every point's score — signal on `v_1`, noise on the rest — and because I have no way
here to tell which of the top `m` directions is the contaminated one (that is the very thing I am trying
to discover), I cannot down-weight the clean directions. The sum dilutes rather than concentrates: it
would help only if I could first make the clean directions *equal in variance* so the contaminated one
stood out, which is a different operation than raw summation and one I have no tool for at this rung. A
second alternative, a full Mahalanobis distance under the class covariance, is worse: the contaminated
covariance already contains the bump, so inverting it down-weights the poison direction — the exact
opposite of what I want. Both alternatives argue *for* the single top eigenvector as the right step now:
it is the one direction I *can* identify as the contaminated one, via the rank-one derivation, without a
clean reference. If it under-catches because the signature is not the loudest direction, the fix is not
to sum more raw directions but to equalize the clean spectrum first — a lever I note and set aside, not
one I can pull with the covariance alone.

Now the computation, ground in this task's scaffold. The harness hands me penultimate features for the
whole training set and the (poisoned) training labels. For each training label, gather that class's
features, compute the class mean `mu_c` (this is `mu_F` per class, the only mean I can actually compute
since I do not know which points are poison), center the features, and take the top *right* singular
vector `v_c` of the centered matrix — that is the top eigenvector of the class covariance, the direction
the rank-one bump inflates. I use the right singular vector, not the left: the centered matrix is
`n x d` with one row per sample, so its right singular vectors are the `d`-dimensional feature-space
directions I project onto. Score each point by the squared centered projection `tau_i = ((r_i - mu_c) .
v_c)^2` — the per-point variance contribution along the contaminated direction. Squaring rather than
absolute value only monotonically reshapes the ranking, so "remove the largest" is unchanged; squaring
is the natural form because the bump I derived is a variance. I do this *per class* because the poison
all sits inside one trained label; if I decomposed the whole dataset at once, ordinary between-class
mean differences would dominate the spectrum and I would just rediscover the labels — the same pooling
mistake I avoided in clustering. The cost of doing this per class is a thin SVD of an `n_c x d` centered
matrix per label — `O(n_c d^2)` for the top singular vector, `d` being 64, 512, or 1,280 — which is cheap
and, unlike the two-means Lloyd loop, carries no distance-concentration dependence at all: I only form
the second-moment matrix and read its top axis.

Let me hand-trace the squared-projection score on a tiny example, both to confirm the poison lands at the
extreme and to watch a subtlety the derivation hides — that `mu_c` is itself contaminated. Collapse a
poisoned class onto the recovered direction: four clean points projecting to `{-1, 0, 0, 1}` and one
poison point at `{5}`. I do not get to center by the clean mean; I center by the class mean I can
actually compute, `mu_c = (-1 + 0 + 0 + 1 + 5)/5 = 1`. Centered values are clean `{-2, -1, -1, 0}` and
poison `{4}`; squared projections are `{4, 1, 1, 0, 16}`. The poison scores `16`, above every clean
point, so removing the top-`1.5*eps` catches it first — the ranking works even though the mean I centered
by is pulled toward the poison. That pull is not free: it drags `mu_c` off the clean center by
`eps*Delta`, which here sharpens the poison's lead, but at a within-class `eps` near `1/2` it would drag
`mu_c` so far that the *clean* points become the extremes — the same `eps < 1/2` cliff the covariance
derivation flagged, now visible in the scoring step itself. This is the concrete reason cifar100 worries
me and cifar10 does not.

One subtlety about *which* label to group by, and it matters more than it looks. The poison's signature
lives in the target class as it was *trained* — the bag of points the network saw as "this class" and
whose representation carries the amplified trigger direction. If I grouped by the model's *predicted*
class at scoring time, a hard poisoned example whose prediction happens to disagree with its assigned
label would be pulled into the wrong group and projected onto a direction fit for a different class,
corrupting its score. So `fit` must remember the training labels and `score_samples(features, logits)`
must apply each point's class-specific direction using those cached labels. The logits the harness
passes are not the grouping signal — they are present only because the interface offers them. This is a
real difference from how the default scaffold (which reads logits) and even the clustering rung treated
the inputs: the spectral test is keyed strictly by training label. A degenerate class with fewer than
two points gets a zero direction rather than a crash.

The harness fixes the removal budget at the largest `1.5*eps` fraction, which is the right instinct and
matches the asymmetry of costs I keep coming back to: only an upper bound on `eps` is known, surviving
poison can preserve the trigger shortcut (the clustering rung's `asr=1.0` is exactly what surviving
poison looks like), and discarding a few extra clean points out of thousands costs little. Over-removing
is the cheaper error, and the harness already does it.

One more piece of the derivation controls how well this works even inside the good regime, and it is the
gap between the contaminated eigenvalue and the largest clean one. The rank-one bump lifts the poison
direction's eigenvalue to about `eps(1-eps)||Delta||^2 + sigma^2`, while the largest *clean* direction
sits at some `lambda_clean`. The estimated top singular vector `v_hat` equals the true `Delta` direction
only if the poison eigenvalue clears `lambda_clean` with room; a standard perturbation bound says the
sine of the angle between `v_hat` and `Delta` scales like `sigma^2 / (eps(1-eps)||Delta||^2 -
lambda_clean)`. Two consequences I can act on. First, when the gap is small the estimate is a *blend* of
the poison direction and the loudest clean direction, and the score projects onto that blend: the poison
projection shrinks by `cos(theta)` and the squared score by `cos^2(theta)`, so a modest misalignment
translates directly into lost recall — this is the mechanism by which "recovered but not cleanly" shows
up as a recall in the low tens of percent rather than near one. Second, and this is the structural
crack, the bound depends on `lambda_clean`: if a class has one big legitimate axis of variation, the
poison must out-shout it to own the top eigenvector, and there is nothing in the single-eigenvector test
that suppresses `lambda_clean`. That is the same "equalize the clean spectrum first" lever the
multi-eigenvector detour pointed at, and it is now doubly motivated — I can see, from the eigengap, that
the top-direction method's recall is capped by however loud the class's largest clean direction happens
to be, a quantity the attacker does not control but the data hands me arbitrarily. I take the single top
eigenvector as this rung's step because it is the one direction the rank-one derivation *licenses* me to
identify without a clean reference, and I read its expected shortfall straight off the eigengap.

So the step-2 delta from clustering is concrete: where 2-means tried to partition high-dimensional
features into compact blobs and collapsed to an arbitrary bisection (recall 0.0016 / 0 / 0, asr pinned
near 1.0), I now read the single contaminated *direction* off each class's covariance spectrum and score
by the squared centered projection onto it. Here is what I expect, falsifiably, against the clustering
numbers. On resnet20-cifar10-badnets, where the BadNets trigger is a strong low-dimensional signal,
recall should rise well off the 0.0016 floor — the spectral bump should be visible where the cluster was
not. On mobilenetv2-fmnist-badnets recall should likewise move above 0. On vgg16bn-cifar100-blend I am
not optimistic the spectral test fully works — the blended trigger is more diffuse and the per-class
poison fraction is higher — so recall may stay near 0 there even with a faithful spectral test. The
honest open question is whether even a clearly-positive recall *removes enough* poison to break the
trigger shortcut: the spectral test ranks the poison direction's extremes, but a single top eigenvector
catches only the poison that aligns with *that one* direction, and if the signature is spread or the
clean covariance is ill-conditioned, recall climbs but `asr` may not collapse. If I see recall improve
yet `asr` stay high, the next rung's job is already named: stop betting on the *single* top direction
and amplify the signature wherever it sits in the spectrum. (The full scaffold module is in the answer.)
