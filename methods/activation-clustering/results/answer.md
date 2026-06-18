# Activation Clustering, Distilled

Activation Clustering detects backdoor-poisoned training rows by analyzing hidden activations of the
suspect model. In a poisoned target class, legitimate target examples and triggered source examples
receive the same label for different internal reasons, so their late-layer activations can form
separate subpopulations. The detector reduces each class's activations to a low-dimensional space,
runs 2-means, then decides which clusters, if any, are poison.

The canonical output is a clean/poison assignment, not a continuous suspicion score:

- `1` means clean / keep the row.
- `0` means poison / remove or relabel the row.

## Core Algorithm

Input: untrusted training set `D_p`, trained suspect model `F`, classes `1..n`.

1. Query `F` on the training samples and collect the last hidden layer activations.
2. Flatten each activation tensor to one vector.
3. Segment activations by class. The paper's Algorithm 1 bins by `F(s)`; ART bins by `y_train`.
4. For each class:
   - reduce dimension, typically to 10 components;
   - run k-means with `k=2`;
   - analyze the two clusters with one of the rules below.
5. Return a report and `is_clean_lst`, with one clean/poison bit per original training row.

Paper experiments use ICA before 2-means and report that 6, 10, and 14 components behave similarly.
ART's `ActivationDefence` object default is `reduce="PCA"` and `nb_dims=10`; its standalone
`cluster_activations` helper defaults to `reduce="FastICA"`.

## Cluster Analysis Cases

**Smaller cluster (`cluster_analysis="smaller"`).**
Mark `argmin(bincount(clusters))` as poison and all other clusters clean. This is ART's object
default. It does not decide that a class is unpoisoned; even a clean class gets its smaller 2-means
split marked suspicious. Ties choose cluster 0 through `argmin`.

**Relative size (`cluster_analysis="relative-size"`).**
Mark clusters whose rounded class fraction is below the threshold. ART defaults to
`size_threshold=0.35`, rounds fractions to two decimals, and uses strict `<`, not `<=`.

**Silhouette (`cluster_analysis="silhouette-scores"`).**
Intended rule: flag a cluster only if it is small enough and the class has a high enough silhouette
score. ART defaults to `size_threshold=0.35` and `silhouette_threshold=0.1`. The paper reports that
thresholds around `0.10` to `0.15` were reasonable, with a trusted clean set preferred for
calibration when available.

**Distance (`cluster_analysis="distance"`).**
For each class, compute the median reduced activation of the whole class and the median of each of
its two clusters. A cluster is suspicious if it is closer to another class's median than to its own
class median, while the sibling cluster is not. This is an ART implementation option, not the paper's
main automatic rule.

**Exclusionary reclassification (`ex_re_threshold > 0`).**
After an analyzer has marked suspicious clusters, train a fresh clone on the rows currently marked
clean. For each suspicious cluster, predict its held-out rows. Let `l` be the number predicted as
the cluster's label and `p` be the number predicted as the most common other class. If `p == 0` or
`l / p > T`, clear the cluster as clean. Otherwise keep it poison, record `l / p` as the ExRe score,
and relabel the rows to the top other class. Equality `l / p == T` remains suspicious in ART.

## Minimal Reference Logic

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import silhouette_score


def reduce_dimensionality(x, nb_dims=10, reduce="FastICA"):
    if x.shape[1] <= nb_dims:
        return x
    if reduce == "FastICA":
        return FastICA(n_components=nb_dims, max_iter=1000, tol=0.005).fit_transform(x)
    if reduce == "PCA":
        return PCA(n_components=nb_dims).fit_transform(x)
    raise ValueError(f"{reduce} dimensionality reduction method not supported")


def assign_clean(clusters, poison_clusters):
    poison_clusters = np.asarray(list(poison_clusters), dtype=int)
    clean = np.ones_like(clusters, dtype=int)
    clean[np.isin(clusters, poison_clusters)] = 0
    return clean


def analyze_smaller(clusters):
    sizes = np.bincount(clusters)
    return assign_clean(clusters, {int(np.argmin(sizes))})


def analyze_relative_size(clusters, size_threshold=0.35, r_size=2):
    sizes = np.bincount(clusters)
    pct = np.round(sizes / float(sizes.sum()), r_size)
    poison = set(np.where(pct < round(size_threshold, r_size))[0])
    return assign_clean(clusters, poison)


def analyze_silhouette(clusters, reduced, size_threshold=0.35, silhouette_threshold=0.1):
    sizes = np.bincount(clusters)
    pct = np.round(sizes / float(sizes.sum()), 2)
    small = set(np.where(pct < round(size_threshold, 2))[0])
    if small and round(silhouette_score(reduced, clusters), 4) > round(silhouette_threshold, 4):
        return assign_clean(clusters, small)
    return np.ones_like(clusters, dtype=int)


def detect_from_activations(activations, labels, reduce="FastICA", analysis="smaller"):
    labels = np.asarray(labels)
    is_clean = np.ones(len(labels), dtype=int)
    report = {}

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        x = reduce_dimensionality(np.asarray(activations[idx]), reduce=reduce)
        clusters = KMeans(n_clusters=2).fit_predict(x)

        if analysis == "smaller":
            clean_bits = analyze_smaller(clusters)
        elif analysis == "relative-size":
            clean_bits = analyze_relative_size(clusters)
        elif analysis == "silhouette-scores":
            clean_bits = analyze_silhouette(clusters, x)
        else:
            raise ValueError("distance and ExRe need class-center/model-retraining state")

        is_clean[idx] = clean_bits
        report[f"Class_{int(cls)}"] = {
            "cluster_sizes": np.bincount(clusters).tolist(),
            "marked_poison": int(np.sum(clean_bits == 0)),
        }

    return report, is_clean.tolist()
```

## Implementation Caveats

The paper-faithful detector uses last-hidden-layer activations. Current ART `main` contains an
active line in `_get_activations` that overwrites `classifier.get_activations(...)` with
`classifier.predict(self.x_train)` despite the nearby comment `# wrong way to get activations`. A
paper-faithful implementation should remove that overwrite.

Current ART `main` also appears to assign `clean_clusters` incorrectly inside the high-silhouette
branch of `analyze_by_silhouette_score`. The intended rule is the one above: when the small cluster
passes both thresholds, that small cluster is poison and the other cluster is clean.
