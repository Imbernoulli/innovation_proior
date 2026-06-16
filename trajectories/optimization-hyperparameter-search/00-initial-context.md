## Research question

Black-box hyperparameter optimization under a tiny budget. Given a model and dataset, find the
configuration that maximizes validation performance, where every function evaluation is a full
train-and-score and the total budget is on the order of *tens* of evaluations (50 for XGBoost, 40 for
SVM and the neural net). The single thing being designed is the **search strategy** — the policy that,
given the search space and the history of past trials, proposes the next configuration to evaluate (and
at what fidelity). Everything else — the benchmarks, the search-space encoding, the evaluation loop, the
metric computation — is fixed. The strategy is called once per loop iteration, sequentially, until the
budget is spent.

## Prior art before the first rung (black-box optimization lineage)

The ladder starts from the simplest defensible thing and climbs toward model-guided, multi-fidelity
search. These are the methods the first rung reacts to and the later rungs combine.

- **Grid search (folklore).** Enumerate a Cartesian product of per-axis values. Coverage is even, but
  the trial count explodes with dimension and every axis gets probed at only a handful of levels — most
  trials are spent varying knobs that do not matter. Gap: wastes the budget on irrelevant axes; infeasible
  past a few dimensions.
- **Random search (Bergstra and Bengio, JMLR 2012).** Draw configurations uniformly from the space. Under
  the empirical fact that the loss has *low effective dimensionality* — only a few hyperparameters matter,
  and which few differs across datasets — random sampling gives every axis as many distinct values as there
  are trials and so covers the important axes far better than a grid. Strong and trivially parallel. Gap:
  **stateless** — every draw ignores every loss already paid for, which under a few-dozen-trial budget is
  the one thing that cannot be afforded.
- **Sequential model-based optimization / Bayesian optimization (Mockus 1978; Jones et al. 1998).** Keep a
  cheap probabilistic surrogate of the expensive loss, and each round pick the next point by maximizing an
  acquisition such as Expected Improvement, which trades exploitation (low predicted loss) against
  exploration (high predictive variance). Gap: the textbook Gaussian-process surrogate needs a metric on the
  whole configuration vector (awkward for categoricals and conditional spaces), costs O(n³) to condition, and
  stakes all exploration on a single point estimate of its own uncertainty that a sparse early sample can
  collapse.
- **Successive halving (Jamieson and Talwalkar, AISTATS 2016).** Evaluate many configurations cheaply (low
  fidelity — fewer epochs/trees/folds), keep the top fraction, and reallocate the saved budget to evaluating
  the survivors at higher fidelity. Gap: fixes one configurations-versus-fidelity tradeoff up front (the
  "n versus B" dilemma), so an aggressive setting can kill a slow-starting but ultimately good configuration.

The ladder picks up exactly where this lineage leaves off: it starts at random search (the stateless
floor), and each rung adds one missing ingredient — adaptation to history, multi-fidelity scheduling, or
both — until they are combined.

## The fixed substrate

A complete HPO loop in `scikit-learn/custom_hpo.py` is frozen and must not be touched. It defines:

- **Three benchmarks**, each a real sklearn tuning problem with a `(space, objective)` pair.
  *XGBoost*: `GradientBoostingRegressor` on California Housing, 6-D (`n_estimators`, `max_depth`,
  `learning_rate` log-scaled, `subsample`, `min_samples_split`, `min_samples_leaf`), budget 50.
  *SVM*: `SVC` on Breast Cancer, 3-D (`C` log, `gamma` log, `kernel` categorical), budget 40.
  *NN*: `MLPRegressor` on Diabetes, 6-D (two log-scaled layer widths, `learning_rate_init` log, `alpha`
  log, `batch_size`, `activation` categorical), budget 40.
- **The fidelity contract.** Each objective accepts `budget ∈ (0,1]`: XGBoost scales `n_estimators` by it
  (floor 10), SVM scales CV folds (`max(2, int(5·budget))`), the NN scales `max_iter` (`max(50, int(500·
  budget))`). Lower fidelity is cheaper and noisier. The loop spends `total_cost += fidelity` each call and
  stops when `total_cost ≥ budget`, so a strategy that evaluates at low fidelity buys *more* evaluations.
- **The search-space API.** `space.params` (list of `HParam` with `name`, `type ∈ {float,int,categorical}`,
  `low`, `high`, `log_scale`, `choices`), `space.dim`, `space.sample_uniform(rng)`, and `space.clip(config)`.
  Each `Trial` records `config`, `score` (validation score, higher is better), and `budget` (fidelity used).
- **The metrics.** `best_val_score` (best score within budget — primary) and `convergence_auc` (trapezoidal
  area under the *min-max-normalized* best-so-far curve plotted against normalized cost — higher means good
  configs were found earlier). Both higher-is-better; reported as the mean over seeds {42, 123, 456}.

## The editable interface

Exactly one region is editable — the `CustomHPOStrategy` class (lines 255–326 of `custom_hpo.py`). The
contract is two methods: `__init__(self, seed=42)` (stores `self.seed` and `self.rng =
np.random.RandomState(seed)`, plus any internal state the strategy needs across calls) and
`suggest(self, space, history, budget_left) -> (config, fidelity)`, which returns the next configuration
dict and a fidelity in `(0,1]`. The loop then clips the fidelity to `[0.1, 1.0]`, clips the config to valid
ranges, evaluates it, appends the `Trial` to `history`, and calls `suggest` again. Because the loop hands
back the full `history` every call, the strategy can carry state either in the object or by re-reading
`history`; either is allowed. Every method on the ladder is a fill of this one contract.

The starting point is the scaffold default: **uniform random search**. Each later method replaces exactly
this class body and nothing else.

```python
# EDITABLE region of scikit-learn/custom_hpo.py (lines 255-326) — default fill
class CustomHPOStrategy:
    """Custom hyperparameter optimization strategy.

    Called repeatedly: suggest(space, history, budget_left) -> (config, fidelity);
    config is evaluated -> score; Trial(config, score, fidelity) is appended to
    history; repeat until budget is exhausted.
    """

    def __init__(self, seed: int = 42):
        """Default: store seed and create RNG. The agent may add internal state."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Default: uniform random search (poor — replace with a better strategy).

        Returns (config, fidelity) with fidelity in (0, 1]; 1.0 = full evaluation.
        """
        config = space.sample_uniform(self.rng)
        return config, 1.0
```

## Evaluation settings

Three benchmarks — XGBoost (6-D, budget 50), SVM (3-D, budget 40), neural net (6-D, budget 40) — each over
three seeds {42, 123, 456}, with mean metrics reported. Two metrics per benchmark, higher is better on
both: `best_val_score` (best validation score within budget) and `convergence_auc` (area under the
normalized best-so-far convergence curve). A method that finds the same final optimum but *earlier* wins on
AUC; a method that wastes budget exploring late loses it.
