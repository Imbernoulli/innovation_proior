## Research question

Black-box hyperparameter optimization under a tiny budget. Given a model and dataset, find the configuration that maximizes validation performance, where each evaluation is a full train-and-score and the total budget is on the order of *tens* of evaluations (50 for XGBoost, 40 for SVM and the neural net). The design object is the **search strategy**: the policy that, given the search space and the history of past trials, proposes the next configuration and the fidelity at which to evaluate it. The benchmarks, search-space encoding, evaluation loop, and metric computation are fixed. The strategy is called once per iteration, sequentially, until the budget is spent.

## Prior art / Background / Baselines

The methods in current use span stateless sampling to model-guided and multi-fidelity search. Each leaves a concrete gap.

- **Grid search.** Enumerate a Cartesian product of per-axis values. Gap: trial count explodes with dimension, and most trials vary knobs that do not matter, so the budget is wasted on irrelevant axes.
- **Random search (Bergstra and Bengio 2012).** Draw configurations uniformly from the space; because the loss has low effective dimensionality, this covers the important axes better than a grid. Gap: it is **stateless** — every draw ignores all loss values already paid for, which is costly under a few-dozen-trial budget.
- **Sequential model-based optimization / Bayesian optimization.** Fit a cheap probabilistic surrogate of the expensive loss and pick each next point by maximizing an acquisition function that trades predicted quality against uncertainty. Gap: the usual Gaussian-process surrogate scales cubically in the number of trials, handles categorical and conditional spaces awkwardly, and can overcommit to an uncertainty estimate formed from sparse early samples.
- **Successive halving (Jamieson and Talwalkar 2016).** Evaluate many configurations at low fidelity, keep the top fraction, and promote the survivors to higher fidelity. Gap: it fixes one configurations-versus-fidelity schedule up front, so an aggressive setting can discard a slow-starting but ultimately good configuration.

## Fixed substrate / Code framework

A complete HPO loop in `scikit-learn/custom_hpo.py` is frozen and must not be touched. It defines:

- **Three benchmarks**, each a real sklearn tuning problem with a `(space, objective)` pair.
  - *XGBoost*: `GradientBoostingRegressor` on California Housing, 6-D (`n_estimators`, `max_depth`, `learning_rate` log-scaled, `subsample`, `min_samples_split`, `min_samples_leaf`), budget 50.
  - *SVM*: `SVC` on Breast Cancer, 3-D (`C` log, `gamma` log, `kernel` categorical), budget 40.
  - *NN*: `MLPRegressor` on Diabetes, 6-D (two log-scaled layer widths, `learning_rate_init` log, `alpha` log, `batch_size`, `activation` categorical), budget 40.
- **The fidelity contract.** Each objective accepts `budget ∈ (0,1]`: XGBoost scales `n_estimators` (floor 10), SVM scales CV folds (`max(2, int(5·budget))`), the NN scales `max_iter` (`max(50, int(500·budget))`). Lower fidelity is cheaper and noisier. The loop spends `total_cost += fidelity` each call and stops when `total_cost ≥ budget`, so low-fidelity evaluations buy more trials.
- **The search-space API.** `space.params` (list of `HParam` with `name`, `type ∈ {float,int,categorical}`, `low`, `high`, `log_scale`, `choices`), `space.dim`, `space.sample_uniform(rng)`, and `space.clip(config)`. Each `Trial` records `config`, `score` (validation score, higher is better), and `budget` (fidelity used).
- **The metrics.** `best_val_score` (best score within budget) and `convergence_auc` (trapezoidal area under the min-max-normalized best-so-far curve versus normalized cost). Both are higher-is-better; reported as the mean over seeds {42, 123, 456}.

## Editable interface

Only the `CustomHPOStrategy` class (lines 255–326 of `custom_hpo.py`) is editable. Its contract is two methods:

- `__init__(self, seed=42)` stores `self.seed` and `self.rng = np.random.RandomState(seed)`, plus any internal state needed across calls.
- `suggest(self, space, history, budget_left) -> (config, fidelity)` returns the next configuration dict and a fidelity in `(0,1]`.

The loop clips fidelity to `[0.1, 1.0]`, clips the config to valid ranges, evaluates it, appends a `Trial` to `history`, and calls `suggest` again. State may live in the object or be inferred from `history`.

The default fill is uniform random search:

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

Three benchmarks — XGBoost (6-D, budget 50), SVM (3-D, budget 40), NN (6-D, budget 40) — each over seeds {42, 123, 456}, with mean metrics reported. `best_val_score` measures final quality; `convergence_auc` rewards finding good configurations earlier.
