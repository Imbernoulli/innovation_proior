## Research question

NAS-Bench-201 holds 15,625 architectures, but a search may only **query 30** of them per dataset
per seed — `NAS_EPOCHS = 30`, hard-enforced (a 31st call to `api.query_val_accuracy` aborts the run).
The single thing being designed is the **search strategy**: given the budget of 30 validation queries,
which 30 architectures do I evaluate, and which one do I return for the (free, unbudgeted) final test
query? Everything else — the search space, the benchmark API, the seeding, the evaluation loop — is
frozen. At K = 30 the algorithm is load-bearing: naïve enumeration is impossible and the spread across
seeds is non-trivial, so the difference between random sampling, evolution, and a learned predictor is
exactly what the metric measures.

## Prior art before the first rung (sample-efficient NAS lineage)

The first rung reacts to the way black-box architecture search was done before any sample-efficiency
pressure, and the methods that tried to compound information across queries are the rungs above it.

- **Grid / manual search over architectures.** Enumerate a Cartesian product of design choices, or
  hand-tune by intuition. Exact in one or two dimensions, but the number of joint configurations is
  exponential in the number of cell edges (5^6 = 15,625 here), so a grid fine enough to resolve the
  important edges is unaffordable, and manual search is irreproducible and unanalyzable. Gap: does not
  scale, no honest baseline.
- **Reinforcement-learning controllers (Zoph and Le, 2017).** An LSTM emits an architecture token by
  token and is nudged by policy gradient toward high-scoring ones. Gets strong numbers but trains a
  whole second network, carries its own schedule and greediness knobs, and couples the workers — heavy
  for a regime where each query is precious and the budget is 30. Gap: complexity and sample cost.
- **GP-based Bayesian optimization (e.g. NASBOT, Kandasamy et al. 2018).** Model the seen `(arch, acc)`
  pairs, pick the next query to maximize expected progress. The right paradigm for expensive black-box
  objectives, but a GP *is* its kernel and there is no off-the-shelf kernel on labeled DAGs — NASBOT
  hand-built an optimal-transport distance — and GP inference is cubic in the number of observations.
  Gap: a bespoke kernel plus cubic cost.

## The fixed substrate

The NAS-Bench-201 cell is frozen: 4 nodes, 6 directed edges, 5 operations per edge
(`skip_connect, none, nor_conv_3x3, nor_conv_1x1, avg_pool_3x3`), so an architecture is a list of 6
integers in `[0, 4]` and there are 5^6 = 15,625 of them. The loop is also frozen: it loads the tabular
benchmark, wraps it in a `BenchmarkAPI` that **counts every `query_val_accuracy` call against the
budget** and raises `BudgetExceededError` past 30, calls `search_step(epoch)` up to 30 times, then
takes `get_best_architecture()` and runs **one unbudgeted test query** to score it. Evaluation is
tabular — no network is trained; a query is a table lookup that costs one unit of budget. Three datasets
(CIFAR-10, CIFAR-100, ImageNet16-120) are three separate settings; seeds `{0,1,2,3,4}`; the metric is
**test accuracy of the returned architecture** (higher is better), reported mean ± std over seeds.

The loop hands the optimizer fixed helpers it may use: `random_architecture()` (a uniform valid arch),
`mutate_architecture(parent)` (a single-edge mutation), `get_neighbors(op_indices)` (all 1-edit
neighbors), `is_valid_arch(...)`, `op_indices_to_arch_str(...)`, and `path_encoding(op_indices)` (a
binary input→output operation-path feature vector of length `5 + 5^2 + 5^3 = 155`, the encoding of
White et al. 2020, useful as predictor input).

## The editable interface

Exactly one region is editable — the `NASOptimizer` class in `naslib/custom_nas_search.py`
(lines 163–234). Every method on the ladder is a fill of this same contract:
`__init__(self, api, num_epochs, seed)` (set up search state), `search_step(self, epoch)` (run one
budgeted step; must call `api.query_val_accuracy` at most once per step and return a metrics dict with
`best_val_acc` and `queries`), and `get_best_architecture()` (return the architecture the harness will
test). The optimizer must maintain `self.best_arch` so the returned architecture is the one it most
wants tested on the unbudgeted split.

The starting point is the scaffold default: **uniform random sampling**, tracking the best-by-validation
architecture seen. Each method on the ladder replaces exactly this class and nothing else.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — default fill
class NASOptimizer:
    """Sample-efficient NAS search strategy (default: uniform random sampling).

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
