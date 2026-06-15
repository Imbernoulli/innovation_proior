**Problem (from step 3).** PaCMAP reached kNN 0.853 / trustworthiness 0.901 on MNIST but plateaus
there, with continuity dipped (0.957) and the *widest* seed spread on the ladder (kNN std 0.0206) — its
graph is three empirically-chosen pair types and its objective rests on hand-tuned working-zone
constants and a hand-scheduled mid-near pull, an empirical construction that wobbles seed to seed and
caps the local fidelity.

**Key idea.** UMAP. Derive both the graph and the objective from principle. A neighbor graph captures
the manifold via the Nerve theorem (cover → Čech complex → graph, truncated to Vietoris–Rips), but a
single ball radius needs uniform data. Assume uniformity and *infer* a per-point Riemannian metric:
demanding equal-population balls forces distance to normalize by each point's own k-neighborhood scale.
Fuzzy membership `exp(−max(0, d − ρ_i)/σ_i)` (ρ_i = nearest-neighbor distance → local connectivity,
σ_i calibrated to a fixed fuzzy cardinality), symmetrized by the probabilistic union
`v_ij = v_{j|i} + v_{i|j} − v_{j|i}v_{i|j}` (edges are existence probabilities). Compare the high- and
low-D graphs by edgewise **cross-entropy** of edge existence, whose `(1−v) log(1−w)` term repels *every*
pair — the symmetric global pressure t-SNE's asymmetric KL lacks — derived, not scheduled.

**Why it works.** Dropping the v-only constant leaves a normalization-free objective, so plain SGD with
edge sampling (attraction) + negative sampling (repulsion) runs linear in edges. The embedding kernel
`(1 + a d^{2b})^{-1}` generalizes t-SNE's Student-t (a=b=1) and is fit to an offset-exponential
carrying `min_dist`. Spectral initialization from the *same* fuzzy graph's normalized Laplacian gives a
globally coherent frame — so the graph serves twice (init + target), cleaner than PaCMAP's
separate-init-plus-three-phase-schedule.

**Scaffold edit / hyperparameters.** Call the `umap` library: `n_components=2`, `n_neighbors=15` (the
manifold-resolution knob, replacing PaCMAP's three counts), `min_dist=0.1` (closeness allowed in the
layout, sets the a,b fit), `metric="euclidean"`, `random_state` from the harness. The library fixes the
connectivity offset ρ, the σ calibration, the union, the a/b curve fit, spectral init, and the
edge-/negative-sampled SGD; the harness exposes n_neighbors, min_dist, and the metric.

**What to watch.** The sharpest prediction is **continuity recovery** — MNIST back up toward ~0.967,
newsgroups toward ~0.91 — since local connectivity guarantees no point is torn off. Trustworthiness
should gain modestly on MNIST (~0.901–0.91) but not jump to 0.96; kNN holds near PaCMAP (~0.84 MNIST);
seed variance should finally tighten (deterministic spectral init). The remaining gap — ~0.96 MNIST
trustworthiness from matching per-point neighbor distributions directly under KL — is what step 5 pays
the O(n^2) cost to close.

```python
class CustomDimReduction:
    """UMAP dimensionality reduction."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import umap
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=self.random_state,
        )
        return reducer.fit_transform(X)
```
