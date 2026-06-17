Let me start from exactly what makes this attack so nasty, because the difficulty is the whole point. Someone has slipped a small number of corrupted images into my training set: they took images, stamped a fixed little trigger on them — a bright pixel in a corner, a faint blended pattern — and relabeled them all to one target class. I train as usual. The resulting model has clean test accuracy that is indistinguishable from a model trained on honest data, but the instant the trigger appears on any input, the model shouts the target class. So I cannot detect anything by looking at accuracy; the validation curve looks pristine. The corruption is invisible from the outside, and it's invisible at the pixel level too — a poisoned target-class image differs from a clean target-class image by a tiny perturbation. My job is to find that small, deliberately camouflaged sub-population sitting inside a much larger, high-variance population of real images, remove it, and retrain without giving the trigger shortcut support. The catch is "small and camouflaged inside high variance," and that catch is what kills the easy ideas.

So let me try the easy ideas first and watch them break, because the failures tell me what the real structure is. The natural framing: within the target class, the training points are a mixture of two sub-populations. A large fraction of clean, correctly-labeled target-class images — call that distribution `D`, fraction `1 - eps` — and a small fraction of corrupted images that really came from some other class but carry the trigger and now wear the target label — call that `W`, fraction `eps`. If `D` and `W` had different enough means, I could just find the direction in which they differ and threshold along it. So the question is: in what space do they differ enough to separate?

First instinct: work at the pixel level. Take all the images labeled with the target class, treat each as a vector, and look for the direction of greatest spread — the top principal direction — hoping the poison sticks out along it. But think about the magnitudes. The trigger is a tiny perturbation; stamping it shifts the mean of the poisoned images away from the clean mean by only a little. Meanwhile the variance across natural images of a class — different objects, poses, colors, backgrounds — is gigantic. So if I project the points onto the top principal direction of the pixel covariance, that direction is governed by the natural image variance, not by the trigger, and the clean and poison projections sit right on top of each other. The mean shift is real but it's drowned. Wall. The pixel space is the wrong space: too much honest variance, too little signal.

Second instinct, cheaper: maybe I don't need a principal direction at all; maybe the poisoned points are just *odd* in some scalar sense. Score every point by the `L2` norm of some representation of it, or by its correlation with a fixed random vector, and flag the extremes. When I imagine the score histograms for clean vs. poison, there is *some* daylight — the poisoned points lean a bit to one side — but the distributions overlap heavily. Any threshold I pick either keeps a chunk of the poison or throws away a lot of clean data. A scalar that ignores the *direction* the poison lives in is too blunt; I'm collapsing a directional signal into a one-number summary and losing exactly the structure that distinguishes the two populations. So weak per-point statistics aren't enough either; I need to find the *right direction* and measure along it.

Now, "find the right direction in which a small contaminating sub-population differs from the bulk, and threshold along it" — that is, almost word for word, the problem that high-dimensional robust statistics has been wrestling with. There, the setup is: I want to estimate the mean of a distribution `D` from samples, but an adversary has replaced an `eps`-fraction of my samples with arbitrary outliers `W`. The hard lesson from that literature (Diakonikolas, Kamath, Kane, Li, Moitra, Stewart; Lai, Rao, Vempala; Charikar, Steinhardt, Valiant) is that coordinate-wise tricks and "throw away points far from the sample mean" lose accuracy that grows with the dimension — useless in feature spaces with hundreds or thousands of coordinates. What works instead is to look at the *covariance spectrum*. And the reason it works is a clean identity I should write down for myself, because everything hinges on it.

Let `F = (1 - eps) D + eps W` be the mixture, with means `mu_D` and `mu_W`, and let `Delta = mu_D - mu_W` be the gap between the two sub-population means. The mixture mean is `mu_F = (1 - eps) mu_D + eps mu_W`. I want the covariance of `F` about `mu_F`, and I want to see where the contamination shows up. Compute the second moment of each sub-population about the *mixture* mean rather than its own mean. For `D`: write `X - mu_F = (X - mu_D) + (mu_D - mu_F)`, and `mu_D - mu_F = mu_D - [(1-eps)mu_D + eps mu_W] = eps (mu_D - mu_W) = eps Delta`. So

  `E_{X~D}[(X - mu_F)(X - mu_F)^T] = Sigma_D + eps^2 Delta Delta^T`,

the cross terms vanishing because `E_{X~D}[X - mu_D] = 0`. Symmetrically, `mu_W - mu_F = (mu_W - mu_D) + ... = -(1 - eps) Delta`, so

  `E_{X~W}[(X - mu_F)(X - mu_F)^T] = Sigma_W + (1 - eps)^2 Delta Delta^T`.

Mix them with weights `(1 - eps)` and `eps`:

  `Sigma_F = (1 - eps) Sigma_D + eps Sigma_W + [(1 - eps)eps^2 + eps(1 - eps)^2] Delta Delta^T`.

That bracket is `eps(1 - eps)[eps + (1 - eps)] = eps(1 - eps)`. So

  `Sigma_F = (1 - eps) Sigma_D + eps Sigma_W + eps(1 - eps) Delta Delta^T`.

There it is. The contamination contributes a *rank-one* bump `eps(1 - eps) Delta Delta^T` sitting on top of the within-population covariances. The mean shift between the two sub-populations announces itself as extra variance, and that extra variance points exactly along `Delta`. So if `Delta` is large enough relative to the within-population spread, the top eigenvector of `Sigma_F` will line up with `Delta`, and projecting onto it separates `W` from `D`. This is the lever the pixel-level attempt was missing — not "the poison is far from the mean" (a scalar idea) but "the poison creates a specific high-variance *direction*."

So why did this fail at the pixel level if it's so clean? Because the bump `eps(1 - eps)||Delta||^2` has to beat the within-population variance to dominate the spectrum, and at the pixel level `||Delta||` is tiny (a trigger barely moves the mean) while `Sigma_D` (natural-image variance) is enormous. The rank-one bump is buried under the leading natural-image directions. So the question becomes: is there a space where `||Delta||` is *large* relative to the within-class variance? Where is the trigger's signal amplified?

The space that can enlarge `Delta` is the network's own representation. Think about what the network is *incentivized* to do during training. The trigger is, by construction, an almost perfect predictor of the target label: nearly every poisoned point has it, and clean points do not. A model minimizing training loss will gladly seize on such a clean, low-variance cue, because keying on the trigger is an easy way to drive down loss on the poisoned points. So the learned representation — the penultimate features — can carry a *strong, dedicated* signal for the trigger, far stronger than the trigger's footprint in raw pixels. The amplification is not incidental; it is the direct consequence of the network being rewarded for using the backdoor. To see that the amplification can be enormous, picture an overparameterized convolutional net with filters to spare: a couple of convolutions can isolate the trigger patch, since it is a fixed pattern that appears almost nowhere else, and then layer after layer can copy that activation and add it to itself, doubling it each time, so the backdoor's contribution to the representation grows exponentially with depth. Backprop need not build exactly that construction, but it shows such representations exist and are exactly the kind the loss pushes toward. Pixels bury the rank-one bump; learned features can lift it above the natural within-class variance.

Now I owe myself the proof, because "should separate" is hand-waving and I want to know *under what condition* this is guaranteed and *how many* I have to remove. Let me set up exactly what "separates" means. Fix `1/2 > eps > 0`. With `F = (1 - eps) D + eps W` and `v` the top eigenvector of `Sigma_F`, say `D` and `W` are *eps-spectrally separable* if there is a threshold `t > 0` such that

  `Pr_{X~D}[ |<X - mu_F, v>| > t ] < eps`   and   `Pr_{X~W}[ |<X - mu_F, v>| < t ] < eps`.

Read it: along `v`, almost no clean point projects beyond `t` (so I rarely discard clean data), and almost no poison point projects within `t` (so thresholding at `t` catches essentially all the poison). If I have this, then removing the points with the largest projection magnitudes onto `v` removes (nearly) all of `W` while sacrificing little of `D`. So the whole game reduces to: prove that the top eigenvector `v` gives spectral separability, and find the condition on `||Delta||` that makes it happen.

Claim: if `Sigma_D, Sigma_W <= sigma^2 I` (within-population variance bounded by `sigma^2` in every direction) and `||mu_D - mu_W||^2 >= 6 sigma^2 / eps`, then `D, W` are eps-spectrally separable. The condition is intuitive already — the mean gap squared has to beat the variance, scaled up by `1/eps` because a smaller poison fraction makes a smaller bump that needs a bigger gap to be visible — but let me actually derive it in three moves.

Move one is just Chebyshev. The variance of `D` along any unit direction `u` is `u^T Sigma_D u <= sigma^2`, so

  `Pr_{X~D}[ |<X - mu_D, u>| > t ] <= sigma^2 / t^2`,

and identically for `W` with `mu_W`. So along any direction, each sub-population concentrates within `~sigma` of its own mean. This is what will let me bound how many clean points stray far and how few poison points stay close — once I know `v` is well-aligned with `Delta`.

Move two: I don't actually need `v` to be exactly `Delta`; I need it correlated enough with `Delta`, but I have to keep the constants honest. Write `c = |<u, Delta>|` and orient `u` so the clean and poison projected means about `mu_F` sit at `eps c` and `-(1 - eps)c`. The recentering identities are the ones I computed: `mu_D - mu_F = eps Delta` and `mu_W - mu_F = -(1 - eps) Delta`. So

  `<X - mu_F, u> = <X - mu_D, u> + eps <Delta, u>`   for `X ~ D`,
  `<X - mu_F, u> = <X - mu_W, u> - (1 - eps) <Delta, u>`   for `X ~ W`.

If I choose a score threshold `t` just outside the clean center, the clean error is controlled by the margin `t - eps c`: Chebyshev gives `Pr_D[|<X - mu_F, u>| > t] <= sigma^2 / (t - eps c)^2` whenever `t > eps c`. The poison error is controlled by the margin between `t` and the poison center: `Pr_W[|<X - mu_F, u>| < t] <= sigma^2 / ((1 - eps)c - t)^2` whenever `t < (1 - eps)c`. This is the correlation-implies-separation calculation. A compact way to state the auxiliary lemma is: if `c > alpha sigma/sqrt(eps)`, then an appropriate shifted threshold puts the clean tail below `eps` and the poison tail on the order of `eps/(alpha - 1)^2`. The eigenvector calculation will at least give the convenient certificate `alpha = sqrt(2)`, and under the `6`-constant hypothesis it actually gives more slack than that. So I keep the margins explicit rather than pretending `sqrt(2)` alone is a sharp threshold: what matters is that the top direction has enough correlation for a threshold to sit between the clean and poison centers with Chebyshev room on both sides.

  `Pr_{X~D}[|<X-mu_F,u>| > t] < eps`,   `Pr_{X~W}[|<X-mu_F,u>| < t] < eps`.

Both bounds shrink as the available gap grows. So I've reduced everything to: how correlated is the *actual* top eigenvector `v` with `Delta`?

Move three: lower-bound `<v, Delta>^2`. Use the covariance identity. From `Sigma_F = (1 - eps)Sigma_D + eps Sigma_W + eps(1 - eps) Delta Delta^T` and dropping the (positive semidefinite) within-population pieces, `Sigma_F >= eps(1 - eps) Delta Delta^T`, so the top eigenvalue obeys `||Sigma_F||_2 >= eps(1 - eps) ||Delta||^2` (test the quadratic form on `Delta / ||Delta||`). Now `v` achieves the top eigenvalue, so

  `eps(1 - eps)||Delta||^2 <= v^T Sigma_F v = (1 - eps) v^T Sigma_D v + eps v^T Sigma_W v + eps(1 - eps) <v, Delta>^2`.

The first two terms are each at most `sigma^2` (bounded covariances), so their convex combination is at most `sigma^2`. Hence

  `eps(1 - eps)||Delta||^2 <= sigma^2 + eps(1 - eps) <v, Delta>^2`,

i.e. `<v, Delta>^2 >= ||Delta||^2 - sigma^2 / (eps(1 - eps))`. Feed in the hypothesis `sigma^2 <= (eps/6)||Delta||^2`:

  `<v, Delta>^2 >= ||Delta||^2 - (eps/6)||Delta||^2 / (eps(1 - eps)) = ||Delta||^2 (1 - 1/(6(1 - eps)))`.

Since `eps < 1/2`, `1 - eps > 1/2`, so `1/(6(1 - eps)) < 1/3`, giving `<v, Delta>^2 > (2/3)||Delta||^2`. And `||Delta||^2 >= 6 sigma^2/eps`, so `<v, Delta>^2 > 4 sigma^2/eps`, which in particular implies the standard weaker certificate `<v, Delta>^2 >= 2 sigma^2/eps` and `|<v, Delta>| >= sqrt(2) sigma/sqrt(eps)`. The important thing is that the top eigenvector is genuinely, quantifiably correlated with the poison direction; with the constant slack needed in move two, the Chebyshev margins give spectral separability. The mean-gap condition `||Delta||^2 >= 6 sigma^2/eps` is the precise statement of "enough amplification," and the `6` is a convenient constant rather than an optimized boundary.

I should sanity-check the regime this assumes, because it tells me when the method *breaks*. The whole derivation needs `eps < 1/2` — the poison must be a minority within the class. If within a class the poison fraction climbs past a half, then the "outlier" sub-population is actually the majority, centering subtracts a mean dominated by poison, and the *clean* images become the ones sitting at the extreme of `v`. So I'd be removing clean data. That's not a flaw in the math; it's the math telling me the operating regime: keep the within-class poison fraction comfortably below 1/2, which in practice means the global poison rate has to be low enough that even after concentrating on one target class the poison doesn't take over that class.

One gap: the lemma is stated for population distributions, but I only have finitely many samples and I compute the *empirical* mean and *empirical* covariance. Does the top eigenvector of the empirical covariance still align with `Delta`? It does, provided I have enough samples to estimate the covariance spectrum accurately and the centered vectors are not heavy-tailed. The tool is a matrix concentration bound: for a random vector `X` with `||X||_2 <= K (E||X||_2^2)^{1/2}` almost surely and second moment `M = E[XX^T]`, the empirical second moment `hatM = (1/n) sum_i X_i X_i^T` from `n` i.i.d. samples satisfies, with high probability, `||hatM - M||_2 <= C (sqrt(K^2 d log d / n) + K^2 d log d / n) ||M||_2`. With `n = Omega(d log n / eps)` samples, Chernoff gives enough clean and poisoned draws in the two empirical sub-populations, and concentration bounds their empirical covariance norms by a small constant factor of `sigma^2`. Pushing the same spectral-separation lemma onto those empirical distributions costs slack: the poison fraction is taken below `1/4`, and the mean-gap condition becomes `||Delta||^2 >= 10 sigma^2/eps`. Then the top eigenvector of the empirical covariance spectrally separates with probability at least `9/10`.

Now the computation is forced. I have the trained network giving me a representation `R(x)` for each input. For each training label `y`, gather the representations of all training points carrying that label, `R(x_1), ..., R(x_n)`. Compute their mean `hatR = (1/n) sum_i R(x_i)` and center: form the `n x d` matrix `M` whose `i`-th row is `R(x_i) - hatR`. The empirical covariance is `(1/n) M^T M`; its top eigenvector is the feature-space direction of greatest spread, and by the identity above that is the direction the poison bump inflates. Equivalently and more cheaply, the top *right* singular vector `v` of `M` is the top eigenvector of `M^T M`, so I take the SVD of `M` and grab `v`. Then score each point by how much it contributes to the variance along that direction:

  `tau_i = ( (R(x_i) - hatR) . v )^2`.

That's the squared projection of the centered representation onto the contaminated direction. Poisoned points, being the source of the rank-one bump, should land at the extreme of this direction and get the largest `tau_i`. I do it class by class, since the poison all sits inside one trained label, so the relevant two-population mixture lives *within* a single label; if I decompose the whole dataset at once, ordinary class-to-class mean differences dominate the spectrum and I just rediscover the labels.

Now, *how many* to remove? Separability guarantees that thresholding at the right `t` catches nearly all poison while dropping few clean points. But I do not know `t`, and the poison budget is only a conservative handle on the scale of the corruption, not an oracle for exactly which points lie on which side of the finite-sample score cutoff. If I remove too narrowly, score overlap or estimation error can leave a few poisoned points in place, and surviving poison can preserve the trigger shortcut. The asymmetry of costs is stark: leaving a few poisoned points in can keep the attack available, whereas discarding a few extra clean points out of thousands should cost little. So I deliberately over-remove: hand the harness scores whose largest `1.5 * eps` fraction is removed. The `1.5` is a safety margin for finite-sample slop and budget uncertainty, at the cost of throwing away a bit of clean data. This is the same instinct the robust-statistics filters use — over-remove to clear the contamination — specialized to a single removal pass instead of an iterate-until-clean loop, because one good representation should already give me enough separation.

Let me also pin down a couple of implementation choices so they're not arbitrary. Top *right* singular vector of the centered matrix `M`, not left: `M` is `n x d` with one row per sample, so the right singular vectors are the `d`-dimensional feature-space directions (eigenvectors of `M^T M`, the `d x d` covariance), which is what I project onto; the left singular vectors are `n`-dimensional and not what I want. The score is the *squared* centered projection: it's the per-point variance contribution along `v`, and squaring (vs. absolute value) only monotonically reshapes the ranking, so for the purpose of "remove the largest" they order points identically — squaring is the natural form because the quantity I derived the bump from is a variance. And I center by the *full* class mean (clean and poison together) before projecting, because `mu_F` is what I can actually compute from the unlabeled-as-poison data, and the whole separability statement was phrased about deviations from `mu_F`.

A subtlety about *which* label to use for grouping. The poison's signature lives in the target class as it was *trained* — i.e., grouped by the poisoned training label, because that's the bag of points the network saw as "this class" and the bag whose representation carries the amplified trigger direction. If instead I grouped by the model's *predicted* class at scoring time, a hard poisoned example whose prediction happens to disagree with its assigned label would get pulled into the wrong group and projected onto a direction fit for a different class, corrupting the score. So `fit` must remember the training labels and `score_samples(features, logits)` must apply each point's class-specific direction using those same cached training labels. The logits are only present because the harness passes them through this interface; they are not the grouping signal.

Let me write it as the procedure I'd actually run:

  Input: trained net `L` with representation `R`, training set, upper bound `eps` on poison fraction.
  For each training label `y`:
    gather `R(x_i)` for all `x_i` with label `y`; let `n = |D_y|`.
    `hatR = (1/n) sum_i R(x_i)`.
    `M = [ R(x_i) - hatR ]_i`        (n x d centered matrix)
    `v = top right singular vector of M`
    `tau_i = ( (R(x_i) - hatR) . v )^2`   for each i
  Return all `tau_i` as suspicion scores.
  The fixed harness removes the largest `1.5 * eps` fraction of scores.
  Retrain `L` from scratch on the surviving points.

And the scoring object that drops into the harness does the per-class SVD in `fit`, caches the training labels, and in `score_samples(features, logits)` applies each cached-label direction to the same training examples. The logits are present because the harness passes them, but the statistic is keyed by the training labels; if the cached labels no longer match the feature matrix, I would rather fail than silently switch to predicted classes and score under the wrong mixture.

```python
import numpy as np


class BackdoorDefense:
    """Per-class representation outlier scoring.

    For each training class c: take the penultimate features of its points,
    center them by the class mean mu_c, take the top right singular vector
    v_c of the centered matrix (= top eigenvector of the class covariance,
    the direction inflated by the rank-one Delta Delta^T poison bump), and
    score each point by the squared centered projection onto v_c:
        tau_i = ((r_i - mu_c) . v_c)^2.
    The harness removes the top ~1.5*eps fraction and retrains.
    """

    def __init__(self):
        self.class_centers = {}      # c -> mu_c  (D,)
        self.class_directions = {}   # c -> v_c   (D,)
        self.cached_labels = None    # training labels from fit()

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.cached_labels = labels.copy()
        for c in np.unique(labels):
            mask = labels == c
            feat_c = features[mask]
            if feat_c.shape[0] < 2:                       # degenerate class: no direction
                self.class_centers[int(c)] = (
                    feat_c.mean(axis=0) if len(feat_c) else np.zeros(features.shape[1])
                )
                self.class_directions[int(c)] = np.zeros(features.shape[1])
                continue
            mu = feat_c.mean(axis=0)                       # mu_c  (= mu_F per class)
            centered = feat_c - mu                         # rows of M
            # Top right singular vector of the centered class matrix = top
            # eigenvector of the class covariance = the contaminated direction.
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self.class_centers[int(c)] = mu
            self.class_directions[int(c)] = vh[0]          # v_c

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        if self.cached_labels is None or len(self.cached_labels) != len(features):
            raise ValueError("score_samples expects the same training examples passed to fit")
        cls_for_sample = self.cached_labels

        scores = np.zeros(len(features), dtype=np.float64)
        for c, v in self.class_directions.items():
            mask = cls_for_sample == c
            if not mask.any():
                continue
            centered = features[mask] - self.class_centers[c]   # r_i - mu_c
            proj = centered @ v                                  # (r_i - mu_c).v_c
            scores[mask] = proj * proj                           # tau_i = (.)^2
        return scores
```

The causal chain is tight now. The attack hides a small, low-pixel-footprint sub-population inside one class, invisible to accuracy and to pixel-level statistics because the trigger's mean shift is drowned by natural-image variance. Modeling a class as a mixture of clean `D` and poison `W` and computing the mixture covariance shows that any mean gap `Delta` between sub-populations creates a rank-one variance bump `eps(1 - eps)Delta Delta^T`, so contamination is a spectral phenomenon and the top eigenvector is pulled toward `Delta`. Pixels fail because `||Delta||` is tiny there; the learned representation can work because the network, rewarded for using the near-perfect trigger cue, amplifies the backdoor signal in feature space, pushing `||Delta||` past the within-class variance. Running the test per training label on centered penultimate features — top right singular vector, score by squared centered projection — is justified by Chebyshev margins plus the eigenvector correlation bound `<v,Delta>^2 >= 2 sigma^2/eps`, with the mean-gap constants `6` in the population statement and `10` in the empirical statement carrying the required slack; a matrix-concentration bound carries the same picture to the empirical covariance with `n = Omega(d log n/eps)` samples under bounded-vector assumptions. Because surviving poison can preserve the trigger shortcut, I over-remove the largest scores as a safety margin. Group by training label so each per-class direction is applied to the points it was fit on, remove the highest-scoring examples under the fixed budget, and retrain from scratch on the filtered set.
