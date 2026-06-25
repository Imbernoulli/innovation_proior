# Spectral Signatures

Spectral Signatures detects backdoor-poisoned training examples by looking for a
minority subpopulation in the covariance spectrum of learned representations.
Train once on the suspect data, score suspicious training examples in feature
space, remove the largest scores, and retrain from scratch.

## Method

For each training label `y`, collect the representation vectors
`r_i = R(x_i)` for examples carrying that label.

1. Compute the class mean `mu_y = mean_i r_i`.
2. Form the centered matrix `M_y` with rows `r_i - mu_y`.
3. Let `v_y` be the top right singular vector of `M_y`.
4. Score each example in the class by

   ```text
   tau_i = ((r_i - mu_y) . v_y)^2.
   ```

5. Remove the top `1.5` times the estimated poison budget from the class.
6. Retrain the network from a fresh initialization on the remaining examples.

The right singular vector is the feature-space principal direction because
`M_y` is sample-by-feature and `M_y^T M_y` is proportional to the empirical
class covariance.

## Derivation

Inside a target label, model the representations as
`F=(1-epsilon)D+epsilon W`, where `D` is the clean class population, `W` is the
poisoned population, and `Delta=mu_D-mu_W`. Then

```text
mu_D - mu_F = epsilon Delta
mu_W - mu_F = -(1-epsilon) Delta
Sigma_F = (1-epsilon)Sigma_D + epsilon Sigma_W
          + epsilon(1-epsilon) Delta Delta^T.
```

The shifted minority population adds a rank-one variance bump along `Delta`.
If `Sigma_D,Sigma_W <= sigma^2 I` and `v` is the top eigenvector of `Sigma_F`,
then

```text
<v,Delta>^2 >= ||Delta||^2 - sigma^2/(epsilon(1-epsilon)).
```

For the actual squared/absolute projection score, a Chebyshev threshold proof
needs the projected mean gap to beat the two-sided margin:

```text
|<v,Delta>| > 2 sigma / ((1 - 2 epsilon) sqrt(epsilon)).
```

Equivalently, one sufficient all-`epsilon<1/2` condition is

```text
||Delta||^2 >
  (1/(1-epsilon) + 4/(1-2epsilon)^2) sigma^2 / epsilon.
```

This is the constant-correct version of the margin argument for the squared
score. The simpler `6 sigma^2/epsilon` population condition
captures the intended low-poison-rate intuition, but its
Chebyshev proof does not establish the absolute-threshold guarantee uniformly
for every `epsilon<1/2`.

## Reference Implementation Check

The MadryLab CIFAR script matches the key direction estimate but not every
Algorithm-1 scoring detail:

- it runs only on the known `target_label`;
- it centers `full_cov` before SVD and takes `v[0:1]`;
- it scores with `norm(v @ full_cov.T)`, an uncentered absolute projection;
- Algorithm 1 scores `((r_i - mu_y).v_y)^2`, the centered squared projection.

The generic implementation below follows Algorithm 1 for all training labels.

## Code

```python
import numpy as np


class BackdoorDefense:
    """Class-conditional Spectral Signatures scorer."""

    def __init__(self):
        self.centers_ = {}
        self.directions_ = {}
        self.labels_ = None

    def fit(self, features, labels, poison_fraction=None, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError("features must have shape (n_samples, n_features)")
        if len(labels) != len(features):
            raise ValueError("labels and features must have the same length")

        self.labels_ = labels.copy()
        self.centers_.clear()
        self.directions_.clear()

        for label in np.unique(labels):
            mask = labels == label
            class_features = features[mask]
            mu = class_features.mean(axis=0)
            centered = class_features - mu

            if len(class_features) < 2 or not np.any(centered):
                direction = np.zeros(features.shape[1], dtype=np.float64)
            else:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                direction = vh[0]

            self.centers_[label] = mu
            self.directions_[label] = direction
        return self

    def score_samples(self, features, logits=None):
        features = np.asarray(features, dtype=np.float64)
        if self.labels_ is None:
            raise ValueError("fit must be called before score_samples")
        if len(features) != len(self.labels_):
            raise ValueError("score_samples expects the same training rows used in fit")

        scores = np.zeros(len(features), dtype=np.float64)
        for label, direction in self.directions_.items():
            mask = self.labels_ == label
            centered = features[mask] - self.centers_[label]
            projection = centered @ direction
            scores[mask] = projection * projection
        return scores

    def filter_indices(self, features, poison_fraction):
        """Return indices kept after per-class 1.5x over-removal."""
        scores = self.score_samples(features)
        remove = []
        for label in np.unique(self.labels_):
            idx = np.flatnonzero(self.labels_ == label)
            k = int(np.ceil(1.5 * poison_fraction * len(idx)))
            if k <= 0:
                continue
            k = min(k, len(idx))
            remove.extend(idx[np.argsort(scores[idx])[-k:]])
        remove = np.asarray(remove, dtype=int)
        keep = np.setdiff1d(np.arange(len(scores)), remove, assume_unique=False)
        return keep
```
