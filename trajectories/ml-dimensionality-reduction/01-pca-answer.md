**Problem.** The scaffold default is a random linear projection — Johnson–Lindenstrauss at two
dimensions, which has nowhere near enough target directions to preserve distances, so it captures an
arbitrary thin slice of the variance. The first real rung needs the cheapest projection that is *not*
arbitrary: directions chosen from the data, symmetric in all coordinates (no privileged dependent
variable, unlike the two regression lines), unique, and ordered.

**Key idea.** Principal Component Analysis. Center the standardized cloud and project onto the
directions of greatest variance. Those directions are the same object three ways — the flat that
minimizes the sum of squared *perpendicular* distances (symmetric total-least-squares fit), the
directions of maximal projected variance (the total scatter splits Pythagoreanly into along + off, the
total is fixed, so minimizing the perpendicular residual is identical to maximizing the projected
variance), and the top eigenvectors of the covariance matrix C (stationarity of the constrained
problem gives `C l = lambda l`, eigenvalue = variance captured). Computed via the SVD of the centered
data, `Xc = U S V^T`, so the right singular vectors are the principal axes and the embedding is the
top columns of `U S`.

**Why it works (and why it is only the floor).** PCA keeps exactly the top-2 variance directions and
discards the rest, with a built-in importance ordering by eigenvalue. It beats the random projection
because the axes are data-driven. But it is *linear*: it can only place points on a flat plane, so a
curved manifold gets folded and points from opposite sides of a fold land adjacent — local
neighborhoods are mangled while the coarse global layout survives. That is the floor's signature
failure and the reason it cannot compete with nonlinear neighbor embeddings.

**Scaffold edit / hyperparameters.** Replace the class body with a call to
`sklearn.decomposition.PCA(n_components=self.n_components, random_state=self.random_state)` — the
sklearn solver *is* this linear-algebra core (center, SVD of the centered matrix, deterministic
`svd_flip` sign convention, projection onto the top components, `random_state` for the randomized
solver). No extra hyperparameters: `n_components=2` and the harness seed.

**What to watch.** kNN accuracy should sit far below the nonlinear methods (around 0.3 on the image
datasets, lower on newsgroups), with trustworthiness dragged down with it; continuity should hold up
relatively, since a linear map preserves the global layout even as it tears local neighborhoods. The
newsgroups data is already linearly pre-reduced by truncated SVD, so a further linear drop to 2D should
be especially weak there. That linearity failure is what forces step 2 to *refine* PCA's global frame
with local nonlinear forces.

```python
class CustomDimReduction:
    """PCA dimensionality reduction (linear baseline)."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        return pca.fit_transform(X)
```
