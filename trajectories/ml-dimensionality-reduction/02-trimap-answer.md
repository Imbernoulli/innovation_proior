**Problem (from step 1).** PCA folds the curved manifold and invents false neighbors — kNN 0.326 /
trustworthiness 0.671 on MNIST — but its high continuity (0.927) shows the global layout is faithful.
The fix must add local nonlinear fidelity *without* throwing away that global frame, and without
falling into the pairwise-target trap that leaves global structure undetermined.

**Key idea.** TriMap. Refine the PCA layout with *relative-order triplets* instead of absolute pairwise
targets. A triplet (i, j, k) — "i closer to j than to k" — is scale-free and carries higher-order
information. Per triplet, minimize a saturating loss `w · s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k))` with the
Student-t kernel `s = (1 + ||·||^2)^{-1}`; in simplified form `l_ijk = w/(1 + d_ik/d_ij)` with
`d_ij = 1 + ||y_i − y_j||^2`. It vanishes once the order is right — so it *stops pulling* satisfied
triplets, which is what lets it refine PCA without tearing the global frame. Weights `w` grow with the
high-D margin `δ_ik^2 − δ_ij^2` (locally density-scaled), tempered by a deformed log to stop giant
margins from dominating. The global structure comes from **PCA initialization**; the triplets supply
only the local fidelity PCA lacks.

**Why it works.** Near-neighbor triplets alone can hit zero loss while scrambling the global layout
(fold the manifold flat — every neighbor still nearest), so they cannot carry global structure. From a
PCA start, far points are already far, so no triplet demands they be pushed apart; the forces only
sharpen local neighborhoods. The saturating loss keeps those forces gentle enough not to disturb the
inherited global frame, and a structured start converges far faster than random noise.

**Scaffold edit / hyperparameters.** Replace the class body with a call to the `trimap` library:
`n_dims=2`, `n_inliers=10` (near-neighbor candidates per point), `n_outliers=5` (random farther points
per inlier → 50 neighbor triplets/point), `n_random=5` (long-range triplets/point, the faint global
insurance term), `n_iters=400` (enough from the PCA start). The library fixes the internal kernel,
local-scale σ (avg distance to 4th–6th neighbors), tempered-log weight transform, PCA init, and
momentum + delta-bar-delta optimizer; the harness exposes only these counts and the iteration budget.

**What to watch.** kNN should jump into the neighbor-embedding regime (MNIST low-mid 0.8s, newsgroups
past 0.6) and trustworthiness climb toward ~0.89, while continuity stays in PCA's high band (~0.958
MNIST) rather than improving — global structure is inherited, not re-derived. It structurally cannot
reach the 0.96 trustworthiness of a direct local-affinity match; that gap motivates step 3.

```python
class CustomDimReduction:
    """TriMap dimensionality reduction."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import trimap
        reducer = trimap.TRIMAP(
            n_dims=self.n_components,
            n_inliers=10,
            n_outliers=5,
            n_random=5,
            n_iters=400,
        )
        return reducer.fit_transform(X)
```
