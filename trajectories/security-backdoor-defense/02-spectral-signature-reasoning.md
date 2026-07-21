The clustering test came back almost exactly as I feared, and the numbers say *why* in detail. On
resnet20-cifar10-badnets `poison_recall` is 0.0016 — of the `1.5*eps` points the harness removed it
caught essentially none of the real poison; on vgg16bn-cifar100-blend and mobilenetv2-fmnist-badnets
recall is a flat 0.0000. And the consequence lands where I said it would: with the trigger shortcut
intact, `asr` stays pinned at 0.97 / 1.00 / 1.00, and `defense_score` (mean 0.4237) is just clean
accuracy carried by the retrain plus a `(1-asr)` term near zero. The defense did nothing; the retrained
model relearned the backdoor from the poison the filter failed to remove.

Read the score formula against these numbers to see where the points come from and where the ceiling is.
`defense_score = 0.5*clean_acc + 0.5*(1-asr)`. On cifar10 the `(1-asr)` half contributed a rounding crumb
(0.0136 out of 0.5) while clean accuracy carried the whole 0.4740; cifar100 scores only 0.3234 because a
100-way problem retrains to 0.6467 clean accuracy with `asr` pinned. So the objective is currently a
clean-accuracy readout with a dead defense term, and the only way to move it is to make `asr` actually
fall, which requires removing enough poison that the retrain forgets the trigger. The clean-accuracy
ceiling per setting (`asr` driven to 0 without hurting clean acc) is `0.5*clean_acc + 0.5`: about 0.96 on
cifar10, 0.82 on cifar100, 0.97 on fmnist. That is the prize, and every point of it is gated behind
detection I do not yet have. This confirms the high-dimensional-clustering collapse I flagged: 2-means on
hundreds of feature dimensions did not find a clean-versus-poison split, it bisected one blob arbitrarily,
so the "smaller cluster" was half of the clean data. Distance-based clustering needs contrast, and in
hundreds of dimensions distances concentrate. So stop asking k-means to *partition* the cloud into two
compact blobs — find the single *direction* the poison shift inflates and measure along it, a direction
the covariance yields even when no compact cluster is visible to a distance metric.

Set up that direction precisely. Within one class the training points are a mixture `F = (1-eps) D +
eps W` of clean points `D` (mean `mu_D`, fraction `1-eps`) and poison `W` (mean `mu_W`, fraction `eps`),
with `Delta = mu_D - mu_W` the gap between the two sub-population means. Compute the mixture covariance
about `mu_F = (1-eps)mu_D + eps mu_W`. For the clean part `mu_D - mu_F = eps Delta`, so its second moment
about `mu_F` is `Sigma_D + eps^2 Delta Delta^T`; for the poison, `mu_W - mu_F = -(1-eps)Delta`, giving
`Sigma_W + (1-eps)^2 Delta Delta^T`. Mixing with weights `(1-eps)` and `eps`, the `Delta Delta^T`
coefficient is `(1-eps)eps^2 + eps(1-eps)^2 = eps(1-eps)`, so

  `Sigma_F = (1-eps) Sigma_D + eps Sigma_W + eps(1-eps) Delta Delta^T`.

The contamination contributes a *rank-one* variance bump `eps(1-eps) Delta Delta^T` on top of the
within-population covariances: the mean shift between clean and poison announces itself as extra variance
pointing exactly along `Delta`. So if that bump is large relative to the within-population spread, the top
eigenvector of `Sigma_F` lines up with `Delta`, and the squared projection onto it separates poison from
clean. I do not need to *group* the points, I need the one direction of anomalous variance, and the
covariance hands it to me directly — and reading a covariance eigenvector does not suffer the
distance-concentration that killed 2-means, because the signal is in the second moment, not in pairwise
distances.

Why would this succeed where the pixel level and, frankly, the clustering both failed? The bump
`eps(1-eps)||Delta||^2` has to beat the within-population variance to dominate the spectrum. At the pixel
level `||Delta||` is tiny while natural-image variance is enormous, so the bump is buried. In the learned
representation the network amplifies the trigger, so `||Delta||` in penultimate-feature space is large,
pushed past the within-class variance — the same amplification I leaned on for clustering, now read off
the covariance rather than handed to a distance metric, which is why it survives the high dimension that
defeated 2-means.

Separation is *guaranteed* under a concrete condition, which is more than I could say for clustering.
Call `D, W` *eps-spectrally separable* if the top eigenvector `v` of `Sigma_F` admits a
threshold `t` with almost no clean point projecting beyond `t` and almost no poison within it; then
removing the largest projections removes nearly all poison while sacrificing little clean data. The
condition: if `Sigma_D, Sigma_W <= sigma^2 I` and `||Delta||^2 >= 6 sigma^2/eps` with `eps < 1/2`, then
`D, W` are eps-spectrally separable — Chebyshev concentration around each sub-population mean, plus the
rank-one bump forcing the top eigenvector to correlate with `Delta` (from `Sigma_F >= eps(1-eps) Delta
Delta^T`, the top eigenvalue is at least `eps(1-eps)||Delta||^2`). The constant `6` is a proof-sketch
constant, not an optimized threshold; the finite-sample version costs a little slack (`eps < 1/4`,
`||Delta||^2 >= 10 sigma^2/eps`, `n = Omega(d log n/eps)` samples).

Put a number to what that demands, using the within-class poison fractions I can work out. On cifar10 the
target class holds ~2,500 poison among ~5,000 clean, within-class `eps ≈ 1/3`, so `||Delta||^2 >=
6 sigma^2/eps` reads `||Delta|| >= sqrt(6/0.33) sigma ≈ 4.2 sigma` — the poison mean must sit more than
four within-population standard deviations off the clean mean along the recovered direction. A stiff bar,
but exactly the one trigger amplification is built to clear: the network is rewarded for making the
trigger a near-perfect, low-variance predictor, precisely a large-mean-shift, small-`sigma`
sub-population. Where amplification is only partial, or where `n` per class is small next to `d`
(CIFAR-100 classes have ~500-1,000 points against `d=512`, badly violating the sample condition), the
guarantee lapses and the recovered direction is polluted by estimation noise. So the arithmetic predicts
a graded outcome: cleanest on the low-`d`, high-`n` cifar10 badnets, shakiest on cifar100 where the
sample budget is thin and the within-class fraction nearest the `1/2` cliff.

That cliff is the regime where the method breaks. The argument needs `eps < 1/2` within the class. If the
within-class poison fraction climbs past a
half, centering subtracts a mean dominated by poison and the *clean* images become the extreme points of
`v` — I would remove clean data. That is why the cifar100-blend setting uses 1% global poison (target
class ~33% poison) rather than 5% (which would make the target class 83% poison, pointing the test at the
wrong sub-population). So on cifar100 I should not expect miracles even from a faithful spectral test —
but I should at least expect it to stop being blind, the bar clustering failed.

The tempting alternative to the single top eigenvector scores by the summed squared projection
onto the top-`m` eigenvectors, on the theory that a spread-out poison shift is caught by more directions.
It falls apart on its own terms: the rank-one bump inflates exactly one direction; the other top
eigenvectors `v_2..v_m` are, by construction, the directions of largest *clean* variance (pose, sub-type,
lighting), each contributing an order-`sigma_clean^2` squared projection that is the same for poison and
clean. Summing them adds `m-1` terms of pure clean spread to every point's score — signal on `v_1`, noise
on the rest — and I have no way to tell which of the top `m` directions is the contaminated one, so I
cannot down-weight the clean ones. The sum dilutes rather than concentrates; it would help only if I
could first make the clean directions equal in variance so the contaminated one stood out, a different
operation I have no tool for here. A full Mahalanobis distance is worse: the contaminated
covariance already contains the bump, so inverting it down-weights the poison direction. Both argue *for*
the single top eigenvector now — the one direction I can identify as contaminated without a clean
reference. If it under-catches because the signature is not the loudest direction, the fix is to equalize
the clean spectrum first, a lever I note and set aside.

Now the computation, on this scaffold. The harness hands me penultimate features and the (poisoned)
training labels. For each training label, gather that class's features, compute the class mean `mu_c`
(which is `mu_F` per class — the only mean I can compute, since I do not know which points are poison),
center, and take the top *right* singular vector `v_c` of the centered matrix — the top eigenvector of
the class covariance, the direction the rank-one bump inflates. Right, not left: the centered matrix is
`n x d` with one row per sample, so its right singular vectors are the `d`-dimensional feature-space
directions I project onto. Score each point by the squared centered projection `tau_i = ((r_i - mu_c) .
v_c)^2` — the per-point variance contribution along the contaminated direction. Squaring only
monotonically reshapes the ranking, so "remove the largest" is unchanged, and it is the natural form
because the bump is a variance. Per class, because the poison sits inside one trained label; decomposing
the whole dataset at once, between-class mean differences would dominate the spectrum and I would
rediscover the labels. The cost is a thin SVD of an `n_c x d` centered matrix per label, `O(n_c d^2)` for
the top vector — cheap, and unlike Lloyd carrying no distance-concentration dependence.

A subtlety about `mu_c`: it is itself contaminated — the poison drags it off the clean center by
`eps*Delta` — but at within-class `eps` below 1/2 the poison still lands at the extreme of the squared
projection and ranks first; only near `eps ≈ 1/2` does the pull drag `mu_c` far enough that the *clean*
points become the extremes. I center by `mu_c` anyway because it is the only mean I can compute without a
poison oracle.

One subtlety about *which* label to group by matters more than it looks. The poison's signature lives in
the target class as it was *trained* — the bag of points the network saw as "this class", whose
representation carries the amplified trigger direction. Grouping by the model's *predicted* class at
scoring time would pull a hard poisoned example whose prediction disagrees with its assigned label into
the wrong group and project it onto a direction fit for a different class. So `fit` remembers the
training labels and `score_samples` applies each point's class-specific direction using those cached
labels; the logits are present only because the interface offers them. A degenerate class with fewer than
two points gets a zero direction rather than a crash.

One more piece controls how well this works inside the good regime: the eigengap between the contaminated
eigenvalue and the largest clean one. The bump lifts the poison direction's eigenvalue to about
`eps(1-eps)||Delta||^2 + sigma^2`, while the largest clean direction sits at `lambda_clean`. A standard
perturbation bound says the sine of the angle between the estimated `v_hat` and `Delta` scales like
`sigma^2 / (eps(1-eps)||Delta||^2 - lambda_clean)`. When the gap is small, `v_hat` is a *blend* of the
poison direction and the loudest clean direction, the poison projection shrinks by `cos(theta)` and the
squared score by `cos^2(theta)` — the mechanism by which "recovered but not cleanly" shows up as partial
recall rather than near one. And structurally, the bound depends on `lambda_clean`: a
class with one big legitimate axis of variation forces the poison to out-shout it to own the top
eigenvector, and nothing in the single-eigenvector test suppresses `lambda_clean`. That is the same
"equalize the clean spectrum first" lever the multi-eigenvector detour pointed at, now doubly motivated —
the top-direction method's recall is capped by however loud the class's largest clean direction happens
to be, a quantity the attacker does not control but the data hands me arbitrarily.

So the prediction: on cifar10's strong low-dimensional BadNets signal recall should rise well off the
0.0016 floor, and on mobilenetv2-fmnist above 0; on cifar100 the diffuse blend and higher per-class
fraction may leave it near 0 even with a faithful test. The open question is whether a clearly-positive
recall *removes enough* poison to break the trigger shortcut — a single top eigenvector catches only
poison aligned with that one direction, so if the signature is spread or the clean covariance
ill-conditioned, recall climbs but `asr` may not fall. If recall improves yet `asr` stays high, the move
is to stop betting on the *single* top direction and amplify the signature wherever it sits in the
spectrum. The full module is in the answer.
