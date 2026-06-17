# Spectral Signatures, distilled

Spectral Signatures is a backdoor-defense method: it detects and removes trigger-poisoned
training examples by spotting them as outliers in the **spectrum of the covariance of the
network's learned representations**, then retrains on the cleaned data. The key observation is
that a backdoor attack leaves a detectable trace — a "spectral signature" — in the learned feature
space, because the network amplifies the trigger signal it is incentivized to use for
classification.

## Problem it solves

An adversary injects an `eps`-fraction of training images that carry a fixed trigger and are
relabeled to a target class. The trained model keeps clean test accuracy but misclassifies any
triggered input as the target. The defender, given the (untrusted) training set, the trained
model, and an upper bound `eps` on the poisoned fraction, must identify and remove the poisoned
points so retraining can proceed on a filtered set while preserving clean accuracy.

## Key idea

Within one class, the training points are a mixture `F = (1 - eps) D + eps W` of clean points `D`
(mean `mu_D`) and poison `W` (mean `mu_W`). With `Delta = mu_D - mu_W`, the mixture covariance is

```
Sigma_F = (1 - eps) Sigma_D + eps Sigma_W + eps(1 - eps) Delta Delta^T,
```

so contamination adds a **rank-one variance bump** along `Delta`. If `||Delta||` is large enough
relative to the within-population spread, the top eigenvector `v` of `Sigma_F` aligns with `Delta`,
and projecting onto `v` separates poison from clean. This fails on raw pixels (the trigger barely
shifts the mean while image variance is huge), but **succeeds on the network's penultimate-layer
representation**: the network, rewarded for keying on the near-perfect trigger cue, amplifies the
backdoor signal, blowing up `||Delta||` in feature space. Run the spectral test **per class** (the
poison lives inside the target class), score each point by its squared projection onto the
contaminated direction, and remove the top scorers.

## Algorithm

```
Input: trained net L with representation R; training set; upper bound eps on poison fraction.
For each training label y:
    Gather R(x_i) for all x_i with label y;  n = |D_y|.
    hatR = (1/n) sum_i R(x_i)                       # class mean
    M    = [ R(x_i) - hatR ]_{i=1..n}               # n x d centered matrix
    v    = top right singular vector of M           # = top eigenvector of class covariance
    tau_i = ( (R(x_i) - hatR) . v )^2               # outlier score (squared centered projection)
Return all tau_i as suspicion scores.
The fixed harness removes the largest 1.5 * eps fraction of scores.
Retrain L from scratch on the surviving points.
```

The `1.5x` over-removal is a deliberate safety margin: only an upper bound on `eps` is known, and
surviving poison can preserve the backdoor, while discarding a few extra clean points out of
thousands is the cheaper error.

## Why it works (separation guarantee)

Define `D, W` to be **eps-spectrally separable** if for the top eigenvector `v` of `Sigma_F` there
is a threshold `t > 0` with `Pr_{D}[|<X - mu_F, v>| > t] < eps` and `Pr_{W}[|<X - mu_F, v>| < t] <
eps` — almost no clean point projects beyond `t`, almost no poison point within it.

**Lemma.** If `Sigma_D, Sigma_W <= sigma^2 I` and `||mu_D - mu_W||^2 >= 6 sigma^2 / eps` (with
`eps < 1/2`), then `D, W` are eps-spectrally separable.

*Proof in three moves.*

1. **Chebyshev:** for any unit `u`, `Pr_{D}[|<X - mu_D, u>| > t] <= sigma^2 / t^2` (and for `W`).

2. **Correlation implies separation:** using `mu_D - mu_F = eps Delta` and `mu_W - mu_F = -(1 -
   eps)Delta`, write `c = |<u, Delta>|`. A threshold `t` separates when it sits between the clean
   offset `eps c` and the poison offset `(1 - eps)c` with enough margin: Chebyshev gives
   `Pr_D[|<X-mu_F,u>|>t] <= sigma^2/(t-eps c)^2` and
   `Pr_W[|<X-mu_F,u>|<t] <= sigma^2/((1-eps)c-t)^2`. Thus a sufficiently correlated direction,
   with the required constant slack, gives both errors `< eps`. The common shorthand
   `|<u,Delta>| >= sqrt(2) sigma/sqrt(eps)` is the spectral-correlation certificate, not by itself
   a strict poison-tail bound under this Chebyshev calculation.

3. **The top eigenvector is correlated with `Delta`:** from `Sigma_F >= eps(1-eps)Delta Delta^T`,
   `eps(1-eps)||Delta||^2 <= v^T Sigma_F v <= sigma^2 + eps(1-eps)<v,Delta>^2`, so `<v,Delta>^2 >=
   ||Delta||^2 - sigma^2/(eps(1-eps))`. With `sigma^2 <= (eps/6)||Delta||^2` and `eps < 1/2`, this
   gives `<v,Delta>^2 > (2/3)||Delta||^2 > 4 sigma^2/eps`, and therefore the weaker stated
   certificate `<v,Delta>^2 >= 2 sigma^2/eps`, i.e. `|<v,Delta>| >= sqrt(2) sigma/sqrt(eps)`.

Move 3 supplies the spectral correlation needed by move 2; the constants are proof-sketch constants,
not optimized thresholds. The condition `||Delta||^2 >= 6 sigma^2 / eps` formalizes "enough
amplification," which is why the method needs the *learned representation* (large `||Delta||`) and a
within-class poison fraction below `1/2` (else the poison becomes the majority and the clean points
become the outliers).

**Finite-sample version.** Via a matrix-concentration bound (for `||X|| <= K(E||X||^2)^{1/2}`
a.s., `||hatM - M||_2 <= C(sqrt(K^2 d log d / n) + K^2 d log d / n)||M||_2` w.h.p.), with
`eps < 1/4`, `n = Omega(d log n / eps)` samples, and the slightly stronger
`||Delta||^2 >= 10 sigma^2/eps`, the top eigenvector of the *empirical* covariance spectrally
separates with probability at least `9/10`.

## Implementation notes

- **Top *right* singular vector of the centered `M`** (`n x d`): these are the `d`-dimensional
  feature-space directions (eigenvectors of `M^T M`), which is what to project onto.
- **Squared centered projection** as the score: the per-point variance contribution along `v`;
  squaring vs. absolute value preserves the centered-projection ranking.
- **Group by the (poisoned) training label**, not the predicted class: the per-class direction is
  only valid for points of the class it was fit on. In the task interface, `fit` caches labels and
  `score_samples(features, logits)` uses those cached labels; logits are ignored except for
  interface compatibility.
- **Released CIFAR code match:** the public `compute_corr.py` filters the target training label,
  centers `full_cov` before SVD, takes `v[0:1]` as the top right singular direction, and scores with
  `eigs @ full_cov.T` followed by an absolute norm/percentile cutoff. The task version keeps the
  same class-conditional centered SVD, runs it for every training label because the target label is
  not exposed, and uses the centered squared Algorithm-1 score
  `tau_i = ((r_i - mu_c).v_c)^2`.

## Working code

```python
import numpy as np


class BackdoorDefense:
    """Per-class Spectral Signatures.

    For each training class c, center its penultimate features by mu_c, take the
    top right singular vector v_c of the centered matrix (top eigenvector of the
    class covariance, the direction inflated by the rank-one Delta Delta^T poison
    bump), and score each point by the squared centered projection
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
            if feat_c.shape[0] < 2:                        # degenerate class: zero direction
                self.class_centers[int(c)] = (
                    feat_c.mean(axis=0) if len(feat_c) else np.zeros(features.shape[1])
                )
                self.class_directions[int(c)] = np.zeros(features.shape[1])
                continue
            mu = feat_c.mean(axis=0)                        # mu_c
            centered = feat_c - mu                          # rows of M
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self.class_centers[int(c)] = mu
            self.class_directions[int(c)] = vh[0]           # top right singular vector v_c

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        if self.cached_labels is None or len(self.cached_labels) != len(features):
            raise ValueError("score_samples expects the same training examples passed to fit")
        cls_for_sample = self.cached_labels                 # apply each point's own class direction

        scores = np.zeros(len(features), dtype=np.float64)
        for c, v in self.class_directions.items():
            mask = cls_for_sample == c
            if not mask.any():
                continue
            centered = features[mask] - self.class_centers[c]
            proj = centered @ v
            scores[mask] = proj * proj                      # tau_i = ((r_i - mu_c).v_c)^2
        return scores
```
