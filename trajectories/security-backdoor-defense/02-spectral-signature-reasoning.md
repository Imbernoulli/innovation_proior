The clustering rung came back almost exactly as I feared, and the numbers say *why* in detail. On
resnet20-cifar10-badnets `poison_recall` is 0.0016 — out of the `1.5*eps` points the harness removed, it
caught essentially none of the real poison; on vgg16bn-cifar100-blend and mobilenetv2-fmnist-badnets the
recall is a flat 0.0000. And the consequence shows up exactly where I said it would: with the trigger
shortcut left intact, `asr` stays pinned at 0.97 / 1.00 / 1.00, and `defense_score` (mean 0.4237) is
just clean accuracy carried by the retrain plus a `(1-asr)` term that is near zero. The defense did
nothing; the retrained model relearned the backdoor from the poison the filter failed to remove. This is
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

This derivation also tells me the regime where the method *breaks*, and it is exactly the regime the
task-design notes warned about. The whole argument needs `eps < 1/2` within the class. If the within-
class poison fraction climbs past a half, the "outlier" sub-population becomes the majority, centering
subtracts a mean dominated by poison, and the *clean* images become the extreme points of `v` — I would
remove clean data. That is why the vgg16bn-cifar100-blend setting uses 1% global poison (keeping the
target class ~33% poison) rather than 5% (which would make the target class 83% poison): at 83% the
spectral test is pointed at the wrong sub-population by construction. So on cifar100 I should not expect
miracles even from a faithful spectral test — but I should at least expect it to stop being blind, which
is the bar the clustering rung failed.

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
mistake I avoided in clustering.

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
