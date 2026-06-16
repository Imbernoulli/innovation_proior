**Problem.** Black-box HPO under a few-dozen-evaluation budget over mixed, log-scaled, partly categorical
spaces. The strategy must propose a valid configuration each call with no gradients and no view inside the
objective. The floor is the strategy that does the minimum defensible thing, so every adaptive method has a
clean yardstick.

**Key idea.** Draw each configuration i.i.d. uniform from the space (uniform-in-log for scale knobs, which
`space.sample_uniform` handles), full fidelity, history ignored. Random — not grid — because aligned grid
points collapse under projection and probe each knob at only the K-th root of the budget, whereas T
independent draws give T distinct values on *every* axis, matching the budget's resolution on whichever
low-dimensional subspace turns out to matter. The hit probability `1 − (1 − v/V)^T` for a relative-volume
target is dimension-free, which is why random search is a strong *final-quality* baseline.

**Why this and not more.** A baseline must isolate one mechanism. Random search isolates "coverage with no
adaptation" — no surrogate, no scheduling, no memory — so any later win is attributable to the one
ingredient that method adds. Its single, deliberate weakness is statelessness: under a tiny budget,
discarding every loss already paid for is the most expensive thing possible, and that cost is what the
ladder exists to remove.

**Hyperparameters.** None beyond the seed. Every call: one `space.sample_uniform(self.rng)` draw at
`fidelity = 1.0`; `history` and `budget_left` deliberately unused.

**What to watch.** Competitive *final* best scores everywhere (low effective dimensionality), but weak and
high-variance *convergence AUC*, worst on the 6-D NN space — because nothing concentrates draws toward where
evidence already points. That is what forces a history-using method at step 2.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — step 1: Random Search
class CustomHPOStrategy:
    """Random Search: sample configurations uniformly at random."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        config = space.sample_uniform(self.rng)
        return config, 1.0
```
