# Activation Clustering, distilled

Activation Clustering (AC) detects backdoor-poisoned training examples by clustering, per class, the
penultimate-layer (last hidden layer) activations of the training data. In a poisoned target class,
the injected samples form a distinct sub-population from the clean samples in feature space — because
the network amplifies the trigger — so two-means on the (dimensionality-reduced) activations splits
the class into a clean cluster and a poison cluster. The poison cluster is the *smaller* one
(adversary poisons under half the class). It needs no trusted/clean dataset and no per-point
retraining, and it flags individual training rows.

## Problem it solves

Given only a (possibly) backdoor-poisoned model and the untrusted training set — no trusted clean
data — identify which individual training examples are the injected poison, so they can be removed
and the model retrained. The attack (BadNets-style) stamps a fixed trigger on source-class images,
relabels them to a target class, and appends them; the model keeps normal clean accuracy and only
misbehaves on triggered inputs, so held-out validation cannot reveal it.

## Key idea

1. The poison is a minority sub-population. It is invisible in pixel space (its variation is buried
   under natural image variance) but separable in the network's learned representation, because the
   trigger is a strong predictor of the target label and the feature extractor is pushed to amplify
   it. So analyze last-hidden-layer activations, not inputs (early layers carry generic low-level
   features that only add noise).
2. Segment activations by class and analyze each class alone — this removes the loud between-class
   variance and leaves clean-vs-poison as the dominant within-class structure.
3. Reduce dimensionality before clustering. Flattened activations are tens/hundreds of thousands of
   dimensions, where Euclidean distance loses contrast and clustering fails. Reduce to ~10
   dimensions. ICA is preferred over PCA: PCA's max-variance axes track natural within-class spread,
   while ICA's independent/non-Gaussian components surface a distinct injected sub-population. Robust
   to using 6/10/14 components.
4. Cluster each class with k-means, k=2 (two compact blobs; faster and at least as accurate as
   DBSCAN/GMM/affinity propagation here).
5. k-means always returns two clusters, so add a poison-vs-clean test:
   - **Relative size:** a poisoned class splits lopsidedly (~p% / (100-p)%, p < 50); a clean class
     splits ~50/50. Lopsided => poisoned; the smaller cluster is poison.
   - **Silhouette score:** high (>~0.10-0.15) when two clusters genuinely fit (poisoned); low when
     one blob was arbitrarily bisected (clean). Take the smaller cluster when high.
   - **Exclusionary reclassification (most accurate; needs a retrain):** retrain without the suspect
     cluster, then classify it. Clean data is classified as its label; poison is reclassified to its
     source class. With l = #classified-as-label and p = #classified-as-the-top-other-class, the ExRe
     score l/p > T (default 1) => clean, < T => poison (and that other class is the source).

In all cases the smaller cluster is the poison cluster; the methods differ only in deciding whether
poison is present.

## Suspicion score (final scoring rule)

Per class, two-means the activations, pick the smaller cluster as poison, and score each sample:

```
score(sample) = 10 * [sample in smaller cluster] + 1 / (||feature - small_centroid|| + 1e-8)
```

The membership term (the constant 10) dominates the ranking, so every flagged-poison point sorts
above every clean point when the harness removes the top-scoring fraction. The closeness term breaks
ties *within* the poison cluster, ranking the densest, most-certain poison (closest to the cluster
core) highest. The harness then removes the top 1.5*epsilon fraction (an over-estimate of the poison
count, for high recall) and retrains. Backdoor repair: relabel flagged poison to its source class
and continue training to convergence (faster than retraining from scratch).

## Full algorithm

```
Input: untrusted training set D_p with classes {1..n}; trained model F.
1. Query F with all of D_p; collect last-hidden-layer activations, flatten each to a 1-D vector.
2. Segment activations by class: A[i] = activations of all samples labeled i.
3. For each class i:
     red      = reduce_dimensionality(A[i])      # ICA (default) or PCA, ~10 dims
     clusters = kmeans(red, k=2)                 # two-means
     analyze_for_poison(clusters)                # relative-size / silhouette / exclusionary-reclass
                                                 # -> smaller cluster is poison if poison present
4. Flag members of poison clusters; remove and retrain (or relabel-to-source and continue training).
```

## Working code

Faithful to the IBM Adversarial Robustness Toolbox `ActivationDefence` (`analyze_by_size`: poison =
the smaller cluster), filling the harness's `fit` / `score_samples` slot.

```python
import numpy as np


class BackdoorDefense:
    """Per-class 2-means activation clustering. The smaller cluster in a class is the
    poison sub-population (adversary poisons < 50% of the class)."""

    def __init__(self):
        self.class_labels_cluster = {}   # cls -> per-sample indicator: 1 if in the smaller cluster
        self.class_small_center = {}     # cls -> centroid of the smaller (poison) cluster
        self.class_indices = {}          # cls -> global sample indices

    def _kmeans2(self, X, max_iter=50):
        # Two-means on the rows of X (two compact blobs: clean vs poison).
        n = len(X)
        c0, c1 = X[0].copy(), X[n // 2].copy()     # spread-out deterministic init
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d0 = np.linalg.norm(X - c0, axis=1)
            d1 = np.linalg.norm(X - c1, axis=1)
            new_labels = (d1 < d0).astype(int)     # 1 if closer to c1, else 0
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            if (labels == 0).any():
                c0 = X[labels == 0].mean(axis=0)
            if (labels == 1).any():
                c1 = X[labels == 1].mean(axis=0)
        n0, n1 = (labels == 0).sum(), (labels == 1).sum()
        small_label = 0 if n0 <= n1 else 1         # minority cluster = poison
        small_center = c0 if small_label == 0 else c1
        return labels, small_label, small_center

    def fit(self, features, labels, poison_fraction, **kwargs):
        features = np.asarray(features, dtype=np.float64)
        labels_arr = np.asarray(labels)
        for cls in np.unique(labels_arr):
            mask = labels_arr == cls
            cls_feat = features[mask]
            indices = np.where(mask)[0]
            self.class_indices[int(cls)] = indices
            if len(cls_feat) < 4:                  # too few to two-means: treat as clean
                self.class_labels_cluster[int(cls)] = np.zeros(len(cls_feat), dtype=int)
                self.class_small_center[int(cls)] = cls_feat.mean(axis=0)
                continue
            clust_labels, small_label, small_center = self._kmeans2(cls_feat)
            self.class_labels_cluster[int(cls)] = (clust_labels == small_label).astype(int)
            self.class_small_center[int(cls)] = small_center

    def score_samples(self, features, logits):
        features = np.asarray(features, dtype=np.float64)
        scores = np.zeros(len(features))
        for cls in self.class_indices:
            indices = self.class_indices[cls]
            is_small = self.class_labels_cluster[cls]
            small_center = self.class_small_center[cls]
            for j, global_idx in enumerate(indices):
                membership_score = float(is_small[j]) * 10.0          # dominant: small-cluster member
                dist = np.linalg.norm(features[global_idx] - small_center)
                closeness = 1.0 / (dist + 1e-8)                       # tie-break: closeness to core
                scores[global_idx] = membership_score + closeness
        return scores
```

The canonical ART version additionally reduces the activations (FastICA / PCA to ~10 dims) before
k-means and supports the relative-size, silhouette, and exclusionary-reclassification analyses to
decide whether a class is poisoned at all; the `smaller`-cluster rule above is its default and is
what the per-sample scoring rule rests on.
