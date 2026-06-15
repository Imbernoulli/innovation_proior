## Research question

I have a few thousand high-dimensional points — MNIST and Fashion-MNIST images as 784-pixel
vectors, 20-Newsgroups documents pre-reduced to 50 TF-IDF/SVD dimensions — and I want a single
two-dimensional embedding that a downstream task can read. The thing being designed is the
**embedding algorithm itself**: how I turn the cloud into 2D so that the neighborhood of every point
survives the projection (who is near whom locally) and the arrangement of the clusters survives too
(how the neighborhoods sit relative to each other globally). I may not use the class labels at fit
time — this is unsupervised; labels only appear afterward, in the kNN-accuracy probe that scores the
map. Everything outside the reducer — how data is loaded, sub-sampled to 5000 points, standardized,
and scored — is fixed.

## Prior art before the first rung (the linear-projection lineage)

The first rung the ladder starts from is the cheapest faithful thing I can do: a linear projection
onto a fixed low-dimensional subspace. That itself is the resolution of an older line of work, and
those ancestors are what the first baseline reacts to.

- **The two regression lines (Galton/Pearson lineage, late 1800s).** Fit a line to a cloud by
  least squares with one variable chosen as the dependent one, and you get a *different* line
  depending on which variable you pick — because the residual is measured along one coordinate axis,
  secretly assuming that axis is error-free. Gap: asymmetric, two answers for one cloud, wrong when
  every coordinate is observed with error.
- **Random projection (Johnson–Lindenstrauss, 1984).** Project onto a random low-dimensional
  subspace; with enough target dimensions pairwise distances are approximately preserved with high
  probability, at zero fitting cost. Gap: at *two* target dimensions there are nowhere near enough
  directions for the guarantee to bite, so a random pair of axes captures an arbitrary, usually tiny,
  slice of the variance — the scaffold default below, and a deliberately poor floor.
- **Factor analysis (Spearman 1904 onward).** Model the observed variables as linear combinations
  of a fixed number of latent factors plus noise. Gap: the number of latent factors is fixed by
  hypothesis, and the solution is rotationally indeterminate — no canonical ordered set of
  directions falls out.

The unresolved demand under all three is a *symmetric*, *ordered*, *unique* linear summary: a small
set of mutually-uncorrelated directions that, in order, capture as much of the cloud's variation as
possible, with no privileged dependent variable and no hypothesized factor count. That is exactly
what the first rung supplies.

## The fixed substrate

The evaluation harness around the reducer is frozen and must not be touched. It loads each dataset,
sub-samples to at most 5000 points, standardizes every feature to zero mean and unit variance
(`StandardScaler`), then constructs one `CustomDimReduction(n_components=2, random_state=seed)`,
calls `fit_transform(X)`, and asserts the output is exactly `(n_samples, 2)` and finite. The same
harness then computes all three metrics at `k=7` neighbors. The reducer receives only the
standardized `X`; it never sees labels. The substrate also fixes the operating envelope: at most
5000 samples, 50 to 784 features, roughly five minutes per dataset on CPU, and only `numpy`,
`scipy`, and `scikit-learn` (plus any neighbor-embedding library a baseline declares) are available.

## The editable interface

Exactly one region is editable: the body of `CustomDimReduction` in
`scikit-learn/bench/custom_dimred.py` (lines 15–59), an `__init__(n_components, random_state)` and a
`fit_transform(X) -> X_reduced` that returns the `(n_samples, n_components)` embedding. Every method
on the ladder is a fill of this same contract — the constructor stores the two settings, and
`fit_transform` is where the entire embedding algorithm lives. There is no separate `fit`/`transform`
split and no out-of-sample requirement: the harness fits and embeds the same `X` in one call, so a
method is free to optimize the 2D coordinates of these exact points directly.

The starting point is the scaffold default: a **random linear projection** — orthonormalized random
axes, the Johnson–Lindenstrauss construction at two dimensions, which is the poor floor each later
method must beat. Each rung replaces exactly this class body and nothing else.

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

Three datasets, each at seeds {42, 123, 456}: **MNIST** (784-d handwritten digits),
**Fashion-MNIST** (784-d grayscale clothing), and **20-Newsgroups** (text reduced to 50-d via
TF-IDF + truncated SVD). Three metrics, all higher-is-better, all at `k=7`:

- **kNN accuracy** — accuracy of a 7-NN classifier trained and tested *in the 2D embedding*
  (stratified shuffle split); measures whether class structure survives the projection.
- **Trustworthiness** — of the 7 embedding-space neighbors of each point, what fraction were also
  near in the original space (penalizes false neighbors the map invents); in [0, 1].
- **Continuity** — of the 7 original-space neighbors of each point, what fraction stay near in the
  embedding (penalizes true neighbors the map tears apart); in [0, 1].
