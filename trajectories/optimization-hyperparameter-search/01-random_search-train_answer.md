The loop hands me `suggest(space, history, budget_left)` and asks for one configuration plus a fidelity, over and over, until the per-benchmark budget (50 for XGBoost, 40 for SVM and the NN) is spent. The whole point of the exercise is the *strategy*, so before I build anything adaptive I need the weakest honest thing to put in that loop — the floor that every later method has to beat. A baseline's job is to make the cost of a missing ingredient legible, and to do that it must isolate exactly one mechanism, with nothing else tangled in. So I want the strategy that does the minimum defensible thing over this mixed, log-scaled, partly categorical domain, and I want it for a reason, not by default.

I propose **uniform random search** as that floor — drawing each configuration i.i.d. uniform from the space, at full fidelity, with the history ignored — and the reason it is *random* and not the more obvious grid is worth stating, because it motivates the whole ladder above it. A grid enumerates the Cartesian product of per-axis levels, so its trial count is exponential in dimension: three levels per axis on the 6-D XGBoost space is $3^6 = 729$ evaluations against a budget of 50, infeasible before it starts. The deeper objection is that a grid's points are *aligned*, so they collapse under projection. With $g$ levels per axis in $K$ dimensions, a grid uses $g^K$ trials but probes each individual knob at only $g = (\text{trial count})^{1/K}$ distinct values, because every grid point shares coordinates with many others. On the subspace of knobs that actually matters, a grid has the resolution of only the $K$-th root of its budget. Refusing to align the points fixes this for free: $T$ independent uniform draws project, almost surely, to $T$ distinct values on *every* axis, so whichever low-dimensional subspace turns out to matter gets the full budget's worth of resolution at once, without my having to know in advance which subspace it is.

There is a clean, dimension-free way to see why this works. Idealize the good region as a target occupying relative volume $v/V$ of the box. Each independent uniform draw misses it with probability $(1 - v/V)$, so the probability that $T$ draws find it is

$$P(\text{hit}) = 1 - \left(1 - \tfrac{v}{V}\right)^{T},$$

and the ambient dimension $K$ does not appear — only the target's relative volume. This is exactly why random search thrives in the regime grid is worst at. Hyperparameter response surfaces empirically have *low effective dimensionality*: on any given dataset only a few knobs move the loss appreciably, and which few differs across datasets, so I cannot just pick the important ones and grid them. A region that is wide along the irrelevant knobs and narrow along the few important ones still has findable relative volume in any number of dimensions. And against precisely those axis-aligned elongated targets, grid is *especially* bad — a thin rectangle either threads through several collinear grid points (redundant, collapsing the effective sample size) or slips entirely between the grid lines (catching nothing) — whereas independent draws are never collinear, so each is an independent shot. This argument (Bergstra and Bengio, 2012) is what makes random search a famously strong baseline on *final* quality.

The scaffold has already done the type-and-scale bookkeeping. `space.sample_uniform(rng)` walks `space.params` and, for each `HParam`, samples categoricals uniformly over `choices`, floats uniformly over $[\text{low}, \text{high}]$ — or uniformly in log space ($\exp$ of a uniform draw over $[\log\text{low}, \log\text{high}]$) when `log_scale` is set — and integers analogously. That log-uniform handling is not incidental: learning rates, layer widths, and the regularizer span orders of magnitude with a roughly flat response per decade, so sampling them uniformly in log space covers the *effect* evenly rather than wasting almost every draw in the top decade of the raw range. Because the scaffold does this correctly, the floor needs to touch no encoding at all — it just calls `sample_uniform`.

Two design knobs remain, and the floor takes the trivial choice on both, deliberately. The first is *fidelity*: the loop accepts a fraction in $(0,1]$ and scales the objective's cost by it, so a cheaper evaluation buys more evaluations against the same budget. The floor returns fidelity $1.0$ on every call — single-fidelity, no cheap noisy partial evaluations — because multi-fidelity is a lever the later rungs will pull, and folding it into the baseline would muddy what the baseline measures. The second knob is *adaptivity*: `suggest` is handed the full `history` every call, so a strategy could read past scores and bias the next draw. The floor ignores `history` and `budget_left` entirely; every draw is i.i.d., independent of everything seen so far. That is the defining property of the floor and the single weakness the entire ladder exists to remove — random search is *stateless*. Under a budget of a few dozen trials, throwing away every loss already paid for is the most expensive thing a strategy can do, and it is exactly what this baseline does on purpose, so that the cost of *not* adapting is what shows up in the numbers. (I keep i.i.d. random rather than a quasi-random Sobol sequence for the same reason: Sobol shaves a little coverage error in low dimensions but sacrifices the i.i.d. structure that makes the baseline anytime-stoppable and analyzable, and buys nothing the ladder cares about.)

The falsifiable expectation is sharp: random search should reach *competitive final best scores* everywhere — the low-effective-dimensionality lesson — but pay for its statelessness in *convergence speed*, especially on the higher-dimensional NN space and in the run-to-run variance of the AUC, because nothing concentrates draws toward where the evidence already points. That diagnosis is already aimed at the next rung: the deficiency is not the final quality of the search but its *use of history*.

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
