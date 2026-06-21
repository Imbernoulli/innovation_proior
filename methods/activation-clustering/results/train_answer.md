A backdoor attack hides poisoned training examples inside an otherwise normal dataset. The adversary takes a small set of source-class inputs, stamps them with a fixed trigger, and relabels them as the target class. After training, the model behaves normally on clean validation data but flips to the target label whenever the trigger appears. This makes the attack invisible to standard validation, and the defender cannot rely on a trusted clean reference set, knowledge of the trigger, or repeated retraining to find the bad rows. The real signal lives inside the model: a successful trigger becomes a strong predictor of the target label, so the late hidden representation should separate poisoned source examples from genuine target examples even when the pixel-level change is tiny.

Earlier ideas fall short for different reasons. Input-space outlier detection fails because natural image variation in pose, lighting, and background can easily swamp a small trigger patch or text signature. Spectral signatures look for the poison direction in the top eigenvector of each class covariance, but that only catches poison aligned with a single dominant direction and can miss diffuse or multi-mode triggers. Fine-pruning repairs models by removing dormant neurons, yet it requires a clean validation set, can hurt clean accuracy, and does not identify the poisoned training rows themselves. What is needed is a detector that works purely from the suspect model and suspect data, uses representation geometry rather than raw pixels, and returns explicit clean-or-poison decisions for every training example.

The method I propose is Activation Clustering. It was introduced by Chen, Liu, Li, Lu, and Song as a defense against hidden-trigger backdoors. The core observation is that, inside a poisoned target class, legitimate target examples and triggered source examples reach the same output label for different internal reasons. The network has learned to route the trigger feature to the target class, so the penultimate-layer activations of the two groups should form separate clusters. Activation Clustering extracts these late hidden activations, analyzes each class separately, reduces the high-dimensional activations to a low-dimensional subspace, runs 2-means clustering, and uses the resulting cluster structure to decide which training rows are poison.

The procedure is straightforward. First, pass the suspect training set through the trained model and collect the activations just before the final classification layer. Flatten each activation tensor into a vector. Next, group the activations by class, either by the model's predicted class or by the training labels stored with the data; the paper's original algorithm uses predicted labels, while the ART implementation defaults to training labels. For each class, reduce the flattened activations to about ten dimensions using ICA or PCA. This step matters because flattened late-layer activations can be tens of thousands of coordinates, and k-means degrades in high dimensions due to distance concentration. Then run k-means with k equal to 2 on the reduced representations. If the class is poisoned, the two clusters should correspond to the legitimate target population and the smaller trigger-bearing population. The simplest rule marks the smaller cluster as poison, relying on the attacker's need to keep poison below half of the class. More conservative rules add a relative size threshold, a silhouette-quality check, or exclusionary reclassification that retrains without the suspicious cluster and checks whether the held-out examples revert to their source class.

The canonical output is a hard clean-or-poison assignment, not a continuous suspicion score. A value of 1 means keep the training example as clean, and 0 means remove or relabel it as poison. Once the poisoned rows are identified, the defender can remove them and retrain from scratch, or relabel them to the recovered source class and continue training. The key advantage is that the entire pipeline requires only the suspect model and the suspect training set, with no trusted clean data and no assumption about what the trigger looks like.

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, PCA
from sklearn.metrics import silhouette_score


def reduce_dimensionality(x, nb_dims=10, reduce="FastICA"):
    """Reduce per-class activations to a low-dimensional subspace."""
    x = np.asarray(x)
    if x.shape[1] <= nb_dims:
        return x
    if reduce == "FastICA":
        return FastICA(n_components=nb_dims, max_iter=1000, tol=0.005).fit_transform(x)
    if reduce == "PCA":
        return PCA(n_components=nb_dims).fit_transform(x)
    raise ValueError(f"Unsupported reduction: {reduce}")


def mark_clusters(clusters, poison_clusters):
    """Return 1 for clean rows and 0 for poison rows."""
    clean = np.ones_like(clusters, dtype=int)
    clean[np.isin(clusters, list(poison_clusters))] = 0
    return clean


def analyze_smaller(clusters):
    """Mark the smaller 2-means cluster as poison."""
    sizes = np.bincount(clusters)
    poison = {int(np.argmin(sizes))}
    return mark_clusters(clusters, poison)


def analyze_relative_size(clusters, size_threshold=0.35):
    """Mark clusters whose rounded fraction is strictly below the threshold."""
    sizes = np.bincount(clusters)
    pct = np.round(sizes / float(sizes.sum()), 2)
    poison = set(np.where(pct < round(size_threshold, 2))[0])
    return mark_clusters(clusters, poison)


def analyze_silhouette(clusters, reduced, size_threshold=0.35, silhouette_threshold=0.1):
    """Mark a small cluster as poison only if the 2-means fit is strong."""
    sizes = np.bincount(clusters)
    pct = np.round(sizes / float(sizes.sum()), 2)
    small = set(np.where(pct < round(size_threshold, 2))[0])
    if small and silhouette_score(reduced, clusters) > silhouette_threshold:
        return mark_clusters(clusters, small)
    return np.ones_like(clusters, dtype=int)


def activation_clustering(activations, labels, reduce="FastICA", analysis="smaller"):
    """
    Activation Clustering backdoor defense.

    Parameters
    ----------
    activations : array-like, shape (n_samples, n_features)
        Last hidden-layer activations flattened to vectors.
    labels : array-like, shape (n_samples,)
        Class labels used to segment activations (typically y_train or predicted labels).
    reduce : {"FastICA", "PCA"}
        Dimensionality reduction used before clustering.
    analysis : {"smaller", "relative-size", "silhouette-scores"}
        Rule for deciding which cluster is poison.

    Returns
    -------
    report : dict
        Per-class cluster sizes and poison counts.
    is_clean : list[int]
        1 = clean, 0 = poison, one entry per input row.
    """
    labels = np.asarray(labels)
    is_clean = np.ones(len(labels), dtype=int)
    report = {}

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        reduced = reduce_dimensionality(np.asarray(activations[idx]), reduce=reduce)
        clusters = KMeans(n_clusters=2, n_init="auto", random_state=0).fit_predict(reduced)

        if analysis == "smaller":
            clean_bits = analyze_smaller(clusters)
        elif analysis == "relative-size":
            clean_bits = analyze_relative_size(clusters)
        elif analysis == "silhouette-scores":
            clean_bits = analyze_silhouette(clusters, reduced)
        else:
            raise ValueError(f"Unknown analysis: {analysis}")

        is_clean[idx] = clean_bits
        report[f"Class_{int(cls)}"] = {
            "cluster_sizes": np.bincount(clusters).tolist(),
            "marked_poison": int(np.sum(clean_bits == 0)),
        }

    return report, is_clean.tolist()


class ActivationClusteringDefence:
    """Poison-filtering interface wrapper around activation_clustering."""

    def __init__(self, classifier, x_train, y_train):
        self.classifier = classifier
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

    def detect_poison(self, reduce="FastICA", analysis="smaller", **kwargs):
        activations = self.classifier.get_activations(self.x_train, layer=-2)
        report, is_clean = activation_clustering(
            activations, self.y_train, reduce=reduce, analysis=analysis
        )
        return report, is_clean
```
