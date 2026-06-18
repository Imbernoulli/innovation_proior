**Problem.** A backdoor is a minority sub-population inside one class: trigger-bearing, relabeled
intruders mixed into the genuine target-class images. The default scaffold scores by max softmax
confidence, which is blind to this — the model fits clean images confidently too, so confidence ranks
poison no better than chance. I need a score that uses the *feature geometry* of the poison
sub-population.

**Key idea.** The network amplifies the trigger (it is a near-perfect predictor of the target), so in
the penultimate-layer activations the poison sits apart from the clean data as its own blob. Segment
the activations by training label, run **2-means** per class, and call the **smaller** cluster the
poison (the attacker poisons < 50% of the class). Score each sample by smaller-cluster membership (the
dominant term, so flagged points sort above clean ones) plus closeness to the small-cluster centroid (a
tie-breaker ranking the densest poison highest).

**Why.** Pixels bury the trigger under natural image variance; the learned representation manufactures
the contrast. Within-class segmentation strips the loud between-class variance so clean-vs-poison is the
dominant remaining structure. The smaller cluster is the poison because poison is the minority.

**What the harness omits / the gamble.** The cleaner form of this method reduces the activations to ~10
dimensions (ICA/PCA) *before* clustering, because k-means degrades in high dimensions (distance
concentration). This scaffold fill clusters the penultimate features **directly, with no learned
reduction** — the self-contained drop-in that matches the `BackdoorDefense` contract. If the two blobs
are well-separated, k-means finds them; if not, the "smaller cluster" is an arbitrary bisection of one
clean blob and recall collapses. That is the floor this rung establishes.

**Hyperparameters.** k=2; deterministic init at rows `0` and `n//2`; up to 50 Lloyd iterations;
membership weight `10.0`; closeness `1/(dist + 1e-8)`; classes with `< 4` samples treated as clean;
float64 throughout. Removal budget `1.5*eps` is fixed by the harness.

```python
# EDITABLE region of custom_backdoor_defense.py — step 1: activation clustering (Chen et al., 2018)
import numpy as np
import torch


class BackdoorDefense:
    """Per-class 2-means activation clustering (Chen et al., 2018).

    For each class, split features into two clusters via k-means.
    The smaller cluster in a class is likely the poisoned sub-population.
    Samples belonging to the smaller cluster get high suspicion scores.
    """

    def __init__(self):
        self.class_labels_cluster = {}   # cls -> array of cluster labels per sample
        self.class_small_center = {}     # cls -> centroid of smaller cluster
        self.class_indices = {}          # cls -> global indices

    def _kmeans2(self, X, max_iter=50):
        """Simple 2-means on rows of X. Returns labels (0 or 1)."""
        n = len(X)
        # Initialise with two spread-out points (deterministic)
        idx0, idx1 = 0, n // 2
        c0, c1 = X[idx0].copy(), X[idx1].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d0 = np.linalg.norm(X - c0, axis=1)
            d1 = np.linalg.norm(X - c1, axis=1)
            new_labels = (d1 < d0).astype(int)   # 0 = closer to c0, 1 = closer to c1
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            mask0 = labels == 0
            mask1 = labels == 1
            if mask0.any():
                c0 = X[mask0].mean(axis=0)
            if mask1.any():
                c1 = X[mask1].mean(axis=0)
        # Determine which cluster is smaller (= poison, minority assumption)
        n0 = (labels == 0).sum()
        n1 = (labels == 1).sum()
        small_label = 0 if n0 <= n1 else 1
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
            if len(cls_feat) < 4:
                # Too few samples to cluster
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
                # Dominant term: membership in the smaller (poison) cluster
                membership_score = float(is_small[j]) * 10.0
                # Tie-breaker: closeness to the small-cluster core
                dist = np.linalg.norm(features[global_idx] - small_center)
                closeness = 1.0 / (dist + 1e-8)
                scores[global_idx] = membership_score + closeness
        return scores
```
