**Problem.** Activation clustering collapsed: 2-means on high-dimensional penultimate features bisected
one clean blob arbitrarily (recall 0.0016 / 0 / 0), so almost no real poison was removed and `asr`
stayed pinned near 1.0. Distance-based clustering loses its contrast in high dimensions. I need the
poison signal without asking a distance metric to find a compact cluster.

**Key idea.** Within a class, poison is a sub-population shifted by `Delta`; the mixture covariance is
`Sigma_F = (1-eps)Sigma_D + eps Sigma_W + eps(1-eps) Delta Delta^T` — contamination adds a **rank-one
variance bump along `Delta`**. So the *top eigenvector* `v_c` of each class covariance aligns with the
poison direction, and the squared centered projection `tau_i = ((r_i - mu_c) . v_c)^2` ranks the poison
at the extreme. Read off the covariance spectrum per class; no clustering, so no distance concentration.

**Why.** The network amplifies the trigger (a near-perfect predictor of the target), making `||Delta||`
large in feature space so the bump dominates the spectrum. Per-class because the poison lives in one
trained label; pooling would just rediscover the labels. Group by the **cached training label**, not the
predicted class, so each per-class direction is applied to the points it was fit on. The harness's
`1.5*eps` over-removal is the right asymmetry: surviving poison preserves the backdoor, a few extra
discarded clean points are cheap.

**What the harness exposes / regime.** The target label is not exposed, so the test runs on *every*
training label. The method needs within-class poison fraction `< 1/2`; the cifar100-blend setting uses
1% global poison (target class ~33% poison) precisely to stay in this regime. The single top eigenvector
catches only poison aligned with that one direction — a known limitation if the signature is spread.

**Hyperparameters.** Top right singular vector of the centered class matrix (`np.linalg.svd`); squared
centered projection as the score; cached training labels for routing; degenerate class (`< 2` points) ->
zero direction; float64. Removal budget `1.5*eps` fixed by the harness.

```python
# EDITABLE region of custom_backdoor_defense.py — step 2: spectral signatures (Tran et al., 2018)
import numpy as np
import torch


class BackdoorDefense:
    """Per-class spectral signatures (Tran et al., NeurIPS 2018).

    For each class c: compute the mean mu_c of its penultimate-layer
    features, center them, take the top right singular vector v_c of
    the centered matrix, and score each sample by
        tau_i = ((r_i - mu_c) . v_c)^2.
    Training labels from fit() route each sample to its class direction.
    """

    def __init__(self):
        self.class_centers = {}       # cls -> mean vector (D,)
        self.class_directions = {}    # cls -> top right singular vector (D,)
        self.cached_labels = None     # training labels from fit()

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.cached_labels = labels.copy()
        for cls in np.unique(labels):
            mask = labels == cls
            cls_feat = features[mask]
            if cls_feat.shape[0] < 2:
                # Degenerate class; zero direction
                self.class_centers[int(cls)] = cls_feat.mean(axis=0) if len(cls_feat) else np.zeros(features.shape[1])
                self.class_directions[int(cls)] = np.zeros(features.shape[1])
                continue
            mu = cls_feat.mean(axis=0)
            centered = cls_feat - mu
            # Top right singular vector of the centered class matrix.
            # np.linalg.svd returns vh with rows = right singular vectors.
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            self.class_centers[int(cls)] = mu
            self.class_directions[int(cls)] = vh[0]

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        # Use TRAINING LABELS (cached in fit) to pick the per-class
        # direction.  Using argmax(logits) would mismatch any sample whose
        # prediction disagrees with its training label.
        if self.cached_labels is not None and len(self.cached_labels) == len(features):
            cls_for_sample = self.cached_labels
        else:
            cls_for_sample = np.asarray(logits).argmax(axis=1)

        scores = np.zeros(len(features), dtype=np.float64)
        for cls, direction in self.class_directions.items():
            mask = (cls_for_sample == cls)
            if not mask.any():
                continue
            centered = features[mask] - self.class_centers[cls]
            proj = centered @ direction
            scores[mask] = proj * proj  # tau_i = (centered . v)^2
        return scores
```
