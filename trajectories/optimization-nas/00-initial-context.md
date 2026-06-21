## Research question

NAS-Bench-201 contains 15,625 architectures, but the search budget is only **30 validation queries** per dataset per seed. The design task is the **search strategy**: given 30 queries, which 30 architectures to evaluate, and which one to return for the final unbudgeted test query? The search space, benchmark API, seeding, and evaluation loop are fixed. At 30 queries, enumeration is impossible and results vary across seeds, so the search strategy is the only lever that matters.

## Prior art / Background / Baselines

- **Grid / manual search.** Enumerate a Cartesian product of design choices, or hand-tune by intuition. Grid search is exact in one or two dimensions; the number of joint configurations grows exponentially with the number of cell edges (5^6 = 15,625 here).

- **Reinforcement-learning controllers (Zoph and Le, 2017).** An LSTM emits an architecture token by token and is updated with policy gradient toward higher validation scores.

- **GP-based Bayesian optimization (NASBOT, Kandasamy et al., 2018).** Model observed `(arch, acc)` pairs and pick the next query to maximize expected progress. NASBOT builds an optimal-transport distance as a kernel over labeled DAGs.

## Fixed substrate / Code framework

The NAS-Bench-201 cell is frozen: 4 nodes, 6 directed edges, 5 operations per edge (`skip_connect`, `none`, `nor_conv_3x3`, `nor_conv_1x1`, `avg_pool_3x3`). An architecture is a list of 6 integers in `[0, 4]`, giving 5^6 = 15,625 candidates.

The loop loads the benchmark and wraps it in a `BenchmarkAPI` that **counts every `query_val_accuracy` call against the budget**, raises `BudgetExceededError` past 30, calls `search_step(epoch)` up to 30 times, then takes `get_best_architecture()` and runs **one unbudgeted test query**. Evaluation is tabular — a query is a table lookup that costs one unit of budget. Three datasets (CIFAR-10, CIFAR-100, ImageNet16-120) and seeds `{0, 1, 2, 3, 4}` define separate settings; the metric is **test accuracy of the returned architecture**, reported mean ± std over seeds.

The loop provides helpers the optimizer may use: `random_architecture()`, `mutate_architecture(parent)`, `get_neighbors(op_indices)`, `is_valid_arch(...)`, `op_indices_to_arch_str(...)`, and `path_encoding(op_indices)` (a binary input→output operation-path feature vector of length `5 + 5^2 + 5^3 = 155`).

## Editable interface

Exactly one region is editable — the `NASOptimizer` class in `naslib/custom_nas_search.py` (lines 163–234). Every search method fills the same contract:

- `__init__(self, api, num_epochs, seed)` sets up search state.
- `search_step(self, epoch)` runs one budgeted step, calls `api.query_val_accuracy` at most once, and returns a metrics dict with `best_val_acc` and `queries`.
- `get_best_architecture()` returns the architecture the harness tests on the unbudgeted split.

The optimizer must maintain `self.best_arch` so the returned architecture is the one it most wants tested.

The starting point is the scaffold default shown below. Each method replaces exactly this class and nothing else.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — default fill
class NASOptimizer:
    """Sample-efficient NAS search strategy (default scaffold).

    Helpers provided by the fixed loop:
        random_architecture()              -> list[int]  (random valid arch)
        mutate_architecture(parent)        -> list[int]  (1-edge mutation)
        get_neighbors(op_indices)          -> list[list[int]]
        is_valid_arch(op_indices)          -> bool
        path_encoding(op_indices)          -> np.ndarray (155-dim path features)
    The budgeted API:
        api.query_val_accuracy(op_indices) -> float   (costs 1 query)
        api.query_count / api.remaining_budget
    """

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed
        self.best_arch = None
        self.best_val_acc = -1.0

    def search_step(self, epoch):
        arch = random_architecture()
        val_acc = self.api.query_val_accuracy(arch)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch
        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        return self.best_arch
```

## Evaluation settings

- Search space: NAS-Bench-201 cell (6 edges, 5 ops, 15,625 architectures).
- Budget: `NAS_EPOCHS = 30` validation queries per dataset per seed, hard-enforced.
- Datasets / settings: CIFAR-10, CIFAR-100, ImageNet16-120.
- Seeds: `{0, 1, 2, 3, 4}`; report mean ± std.
- Metric: test accuracy of the returned architecture (one unbudgeted final query); higher is better.
