**Problem (from step 4).** Across TriMap, PaCMAP, and UMAP, trustworthiness has been stuck at ~0.90 on
MNIST — three different methods, same cap. Each spends real capacity on global structure (PCA-inherited
frame / mid-near skeleton / all-pairs symmetric repulsion), and that allocation is itself what caps the
local score. The strongest local result needs a method that spends almost all its capacity matching
local neighborhoods exactly.

**Key idea.** t-SNE. Match per-point neighbor *distributions* directly. High-D side: a Gaussian per
point, bandwidth σ_i set by fixing **perplexity** (effective neighbor count), symmetrized into a joint
`p_ij = (p_{j|i} + p_{i|j})/(2n)` (anchors outliers). Low-D side: a heavy-tailed Student-t
`q_ij ∝ (1 + ||y_i − y_j||^2)^{-1}` — because we match probabilities, the map kernel may differ from the
input Gaussian, and the slow tail lets moderate similarities map to large distances, fixing crowding.
Minimize one `KL(P||Q)`; its asymmetry punishes placing truly-near points far apart, tilting hard toward
local fidelity. Gradient: `4 Σ_j (p_ij − q_ij)(y_i − y_j)(1 + ||y_i − y_j||^2)^{-1}`.

**Why it works (and its cost).** The KL is matched to the *exact* per-point neighborhoods, so local
structure is preserved precisely rather than approximately — the cap the graph methods hit. The price is
the all-pairs partition function in q_ij: O(n^2) per iteration, the cost UMAP's whole design avoided.
Here n ≤ 5000 per dataset, so O(n^2) fits the 5-minute CPU budget — paying it is exactly what lifts the
local ceiling. PCA init + early exaggeration (transiently scale up p_ij so true clusters condense with
room to rearrange) supply the global scaffolding without a hand-tuned schedule.

**Scaffold edit / hyperparameters.** Call `sklearn.manifold.TSNE`: `n_components=2`, `perplexity=30.0`
(the local-scale knob), `learning_rate="auto"` (scaled from n), `init="pca"` (structured global start),
`n_iter=1000`, `random_state` from the harness. The sklearn solver fixes the σ binary search, the
symmetrization, the Student-t kernel, and the early-exaggeration schedule.

**What clears the bar.** Trustworthiness should finally break the ~0.90 plateau — MNIST → ~0.96,
Fashion-MNIST → ~0.98, newsgroups → ~0.94 — with kNN rising too (MNIST → ~0.86, Fashion → ~0.79). The
risk is continuity/global structure: the asymmetric KL has no symmetric repulsion, so continuity should
land roughly level with UMAP's recovered ~0.967 (held up by PCA init + exaggeration), not collapse. If
trustworthiness does not break 0.90, the global/local capacity thesis is wrong. This is the endpoint: no
published neighbor-embedding method clearly beats a direct per-point KL match on this task's k=7 local
metrics within the CPU/5-min/sklearn budget.

```python
class CustomDimReduction:
    """t-SNE dimensionality reduction."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=30.0,
            learning_rate="auto",
            init="pca",
            random_state=self.random_state,
            n_iter=1000,
        )
        return tsne.fit_transform(X)
```
