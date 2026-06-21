## Research question

I have a few thousand high-dimensional points — MNIST and Fashion-MNIST images as 784-pixel vectors, and 20-Newsgroups documents pre-reduced to 50 TF-IDF/SVD dimensions — and I want a single two-dimensional embedding that a downstream task can read. The object being designed is the embedding algorithm itself: how to turn the cloud into 2D so that local neighborhoods and global cluster arrangement survive the projection. Labels are unavailable at fit time; this is unsupervised. Labels appear only in the kNN-accuracy probe that scores the map. Everything outside the reducer — data loading, sub-sampling to 5000 points, standardization, and scoring — is fixed.

## Prior art / Background / Baselines

The relevant prior methods are:

- **Least-squares regression lines (Galton/Pearson).** A straight line is fit to a cloud by minimizing squared residuals along one coordinate, with one variable chosen as dependent.
- **Random projection (Johnson–Lindenstrauss).** Points are projected onto a random low-dimensional subspace; with enough target dimensions, pairwise distances are approximately preserved with high probability.
- **Factor analysis (Spearman).** Observed variables are modeled as linear combinations of a fixed number of latent factors plus noise.

The scaffold default is a random linear projection: orthonormalized random axes, the Johnson–Lindenstrauss construction at two dimensions, used as the poor floor each method must beat.

## Fixed substrate / Code framework

The evaluation harness around the reducer is frozen. It loads each dataset, sub-samples to at most 5000 points, standardizes every feature to zero mean and unit variance (`StandardScaler`), constructs one `CustomDimReduction(n_components=2, random_state=seed)`, calls `fit_transform(X)`, and asserts the output is exactly `(n_samples, 2)` and finite. The harness then computes all three metrics at `k=7` neighbors. The reducer receives only the standardized `X`; it never sees labels. The substrate fixes the operating envelope: at most 5000 samples, 50 to 784 features, roughly five minutes per dataset on CPU, and only `numpy`, `scipy`, `scikit-learn`, and any neighbor-embedding library a baseline declares are available.

## Editable interface

Only one region is editable: the body of `CustomDimReduction` in `scikit-learn/bench/custom_dimred.py` (lines 15–59). It must provide `__init__(n_components, random_state)` and `fit_transform(X) -> X_reduced` returning an `(n_samples, n_components)` embedding. Every method fills this same contract. There is no separate `fit`/`transform` split and no out-of-sample requirement; the harness fits and embeds the same `X` in one call, so a method may optimize the 2D coordinates of these exact points directly.

The starting point is the scaffold default below: a random linear projection. Each rung replaces exactly this class body and nothing else.

```python
# EDITABLE region of scikit-learn/bench/custom_dimred.py (lines 15-59) -- default fill
class CustomDimReduction:
    """Custom dimensionality reduction method.

    Must implement fit_transform(X) -> X_reduced.
    """

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reduce dimensionality of X to (n_samples, n_components)."""
        # Default: random projection (poor baseline) -- replace with your design
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        projection = rng.randn(n_features, self.n_components)
        projection /= np.linalg.norm(projection, axis=0, keepdims=True)
        X_reduced = X @ projection
        return X_reduced
```

## Evaluation settings

Three datasets, each at seeds {42, 123, 456}: **MNIST** (784-d handwritten digits), **Fashion-MNIST** (784-d grayscale clothing), and **20-Newsgroups** (text reduced to 50-d via TF-IDF + truncated SVD). Three metrics, all higher-is-better, all at `k=7`:

- **kNN accuracy** — accuracy of a 7-NN classifier trained and tested in the 2D embedding (stratified shuffle split); measures whether class structure survives the projection.
- **Trustworthiness** — of the 7 embedding-space neighbors of each point, what fraction were also near in the original space; penalizes false neighbors the map invents.
- **Continuity** — of the 7 original-space neighbors of each point, what fraction stay near in the embedding; penalizes true neighbors the map tears apart.
