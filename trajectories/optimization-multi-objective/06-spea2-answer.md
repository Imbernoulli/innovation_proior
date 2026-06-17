**Problem.** NSGA-III converges well but its diversity is only as even as the *fixed reference grid* it covers,
which is a proxy for the actual point-cloud density the Spread metric scores — so its DTLZ2 spread (0.621) is
barely better than NSGA-II's, and worse than RVEA's. No prior rung measures diversity as a true metric density
of the real solution cloud.

**Key idea.** Strength-Pareto fitness + k-NN density. Give every individual a strength `S(i)` = how many it
dominates, and raw fitness `R(i)` = sum of its dominators' strengths (`R=0` marks the front, graded deeper
into the interior). Break ties with a true k-NN density `D(i) = 1/(sigma_i^k + 2)`, `k = sqrt(N+N_archive)`,
over *actual* objective-space distances; the `+2` keeps `D in (0,1)` so `F = R + D` is layered (dominance
first, real density second). A fixed-size archive copies all nondominated, fills underflow by best-`F`, and on
overflow truncates the densest nearest-neighbour pair — which preserves the extremes automatically.

**Why it tops the ladder.** It optimizes the very quantity Spread measures (real inter-point distances) instead
of grid coverage, so it should beat NSGA-III on spread on every problem while holding convergence at
NSGA-III's level — a clean spread win at no convergence cost. The price is wall-clock: full pairwise distance
matrices each generation make it the slowest rung.

**Why (harness specifics).** `survive` delegates to DEAP's `selSPEA2` (the full strength + k-NN density +
boundary-preserving truncation), then refreshes the archive to the surviving nondominated front. Mating is a
binary tournament on the archive by *dominance* (not full `F`-fitness). SBX crossover probability is back to
**0.9** (vs the reference-point rungs' 1.0).

**Hyperparameters.** Archive size = `pop_size`; `k = sqrt(combined pool)` for density; SBX `eta_c = 20` at
p=0.9, polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 6: SPEA2
class CustomMOEA:
    """SPEA2: Strength Pareto Evolutionary Algorithm 2."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []

    def select(self, population, k):
        """Binary tournament selection using SPEA2 fitness from archive."""
        # Use archive for selection if available, otherwise population
        pool = self.archive if self.archive else population
        # Binary tournament on dominance
        selected = []
        for _ in range(k):
            i1, i2 = random.sample(range(len(pool)), 2)
            a, b = pool[i1], pool[i2]
            if a.fitness.dominates(b.fitness):
                selected.append(deepcopy(a))
            elif b.fitness.dominates(a.fitness):
                selected.append(deepcopy(b))
            else:
                selected.append(deepcopy(random.choice([a, b])))
        return selected

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population, offspring):
        """SPEA2 survival: strength fitness + kNN density truncation."""
        combined = population + offspring

        # Use DEAP's built-in SPEA2 selection
        selected = tools.selSPEA2(combined, self.pop_size)

        # Update archive with non-dominated solutions
        nd = get_nondominated(selected)
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]

        return selected

    def on_generation(self, gen, population):
        pass
```
