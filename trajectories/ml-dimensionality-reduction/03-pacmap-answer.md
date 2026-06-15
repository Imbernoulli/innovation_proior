**Problem (from step 2).** TriMap reached kNN 0.832 / trustworthiness 0.890 on MNIST but plateaus
there, with the widest seed spread on the ladder — because its global structure is *inherited from the
PCA init*, not produced by its triplets (remove the random triplets, no change; remove PCA init, global
structure collapses). A method whose global layout is a hostage to a deterministic start cannot correct
that layout, so the residual frame errors are baked in.

**Key idea.** PaCMAP. Put the global skeleton *into the objective* with a third pair type. Comparing
methods by their forces gives six principles a good local loss obeys, and a *separable* edge loss with
two nonnegative unimodal-vanishing forces satisfies all six — so triplets are unnecessary. But the
principles only buy *local* structure: a thought experiment (and a direct computation of t-SNE's
1/d^3-decaying repulsion) shows no force ever acts on *moderate-distance* pairs, and relative distances
among far things *are* the global structure. So add **mid-near pairs** — attracted, at moderate
distance — alongside neighbors (local) and far pairs (repulsion/crowding). Three plain-edge pair types.

**Why it works.** The loss is three separable rational terms in `d̃ = ||y_i − y_j||^2 + 1`:
`d̃/(10+d̃)` for neighbors (narrow attractive working zone), `d̃/(10000+d̃)` for mid-near (same shape,
wide zone to organize global structure), `1/(1+d̃)` for far pairs (repulsion). A three-phase weight
schedule builds global structure first (mid-near weight annealed from 1000 → 3), stabilizes, then zeros
the mid-near pull and refines local structure — coarse-to-fine, so neighbors are never flung past their
saturated attraction. PCA init is now just a head start, not the source of global structure, so the
layout is no longer a hostage to it.

**Scaffold edit / hyperparameters.** Call the `pacmap` library: `n_components=2`, `n_neighbors=10`,
`MN_ratio=0.5` (→ 5 mid-near pairs/point), `FP_ratio=2.0` (→ 20 far pairs/point), `random_state` from
the harness. The library fixes the scaled-distance neighbor selection (σ = avg distance to 4th–6th NN),
the second-of-six mid-near draw, the three force constants (10 / 10000 / 1), the three-phase schedule,
small PCA init, and the Adam optimizer; the harness exposes only the four counts.

**What to watch.** kNN should edge above TriMap on the image datasets (MNIST → ~0.85) and
trustworthiness nudge up (~0.90 MNIST) — this buys *robustness* of the global structure, not a new
local ceiling. Continuity may dip slightly below TriMap's band (phase-three repulsion carves boundaries
harder), the signature of trading a little continuity for cleaner separation. Still out of reach: the
0.96 trustworthiness of a principled neighbor graph + cross-entropy, which step 4 tests.

```python
class CustomDimReduction:
    """PaCMAP dimensionality reduction (SOTA)."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        import pacmap
        reducer = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=self.random_state,
        )
        return reducer.fit_transform(X)
```
