**Problem.** Random search caught the good region by volume but had no memory: when a seed drew poorly
it could not recover, so the result was governed by luck and the per-seed variance was on the order of
any gap worth opening. The lever is to **exploit the best-seen** — spend later queries near good draws
instead of uniformly.

**Key idea (REA, low-budget variant).** Steady-state evolution under the K = 30 budget. Seed a
population of P = 10 with random architectures (queries 1–10), then for each remaining query run a
tournament of S = 3, mutate the best of the sample by one edge (`mutate_architecture`), evaluate the
child, append it, and **evict the oldest** member. Age-based eviction (pop index 0) gives every cell a
fixed lifespan so a strong cell persists only by being re-discovered — the regularizer that keeps the
~20 evolution steps spread across the space instead of homesteading the first good region. Track the
best-ever separately and return it.

**Why these knobs.** Budget arithmetic: 10 seeds leave ~20 evolution steps; P = 10 holds diversity yet
turns over fast; S = 3 of 10 is moderate greed (the takeover time `~1/ln S` floods P in a few cycles —
fast enough to exploit a good seed, loose enough not to freeze on a bad one). Single-edge mutation is the
smallest local move and keeps the space connected; no crossover (no gain over chained single edits at
this budget). Validity is re-rolled before spending a query.

**What to watch.** Expect the win where random search failed to recover — the wide-spread,
low-saturation setting (ImageNet16-120 should rise above 44.57). Modest lift or wash on CIFAR-100;
on near-saturated CIFAR-10 (random search ≈ 93.38) the narrow-band ceiling may leave REA at or just
below the floor. If REA *loses* on ImageNet16-120, the diagnosis flips: single-edge climbing from random
seeds cannot reach the good region in 20 steps, and the next rung must *model* the surface, not mutate it.

**Hyperparameters.** `population_size = 10`, `tournament_size = 3`, single-edge mutation, kill-oldest;
one query per step; 30 steps.

```python
# EDITABLE region of naslib/custom_nas_search.py (lines 163-234) — step 2: REA
class NASOptimizer:
    """REA — Regularized Evolution Algorithm for NAS (low-budget variant).

    Population size 10 and tournament size 3, tuned for K=30 queries
    following NAS-Bench-Suite (White et al., 2022) low-budget recipes.
    """

    def __init__(self, api, num_epochs, seed):
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self.population_size = 10
        self.tournament_size = 3
        self.population = []  # list of (arch, val_acc)
        self.best_arch = None
        self.best_val_acc = -1.0

    def _update_best(self, arch, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = list(arch)

    def search_step(self, epoch):
        if epoch < self.population_size:
            # Seed initial population with random architectures
            arch = random_architecture()
            val_acc = self.api.query_val_accuracy(arch)
            self.population.append((arch, val_acc))
        else:
            # Tournament selection
            k = min(self.tournament_size, len(self.population))
            sample_indices = random.sample(range(len(self.population)), k)
            parent_idx = max(sample_indices, key=lambda i: self.population[i][1])
            parent_arch = self.population[parent_idx][0]

            # Mutation
            child_arch = mutate_architecture(parent_arch)
            while not is_valid_arch(child_arch):
                child_arch = mutate_architecture(parent_arch)
            child_val_acc = self.api.query_val_accuracy(child_arch)

            # Add child and remove oldest (regularization)
            self.population.append((child_arch, child_val_acc))
            self.population.pop(0)
            arch, val_acc = child_arch, child_val_acc

        self._update_best(arch, val_acc)

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "population_size": len(self.population),
        }

    def get_best_architecture(self):
        return self.best_arch
```
