**Problem.** RVEA recovered 3-objective convergence but at the cost of *instability*: a hard-wired
`max_gen = 400` that mistimes the APD ramp on shorter runs, an un-normalized angle penalty that let DTLZ1
seed 42 blow up (spread 3.45, IGD 0.110), and an adaptation schedule that can churn niches. Every one of
those is a fragile diversity knob. I want the convergence/diversity balance with *no* such parameter.

**Key idea.** NSGA-II: elitist survival with parameter-free diversity. Merge parents and offspring into one
`2N` pool (elitism without an external archive), fast-non-dominated-sort it in `O(M N^2)`, and fill the next
generation front by front. Crowding distance — summed normalized neighbour gaps per objective, with the front
extremes pinned at infinity — measures local crowding with no radius and no horizon, so diversity has zero
free parameters. One lexicographic rule, rank then crowding, drives both mating (binary DCD tournament) and
survival (truncate the critical front by crowding).

**Why it ranks above RVEA.** Not by converging better — RVEA matched it on DTLZ2 — but by being *stable*:
no horizon to mistime, no un-normalized penalty to let a niche poach, so per-seed numbers stay tight and the
DTLZ1 blow-up seed should disappear. It wins the 2-objective ZDTs on HV and spread. Its one weakness is the
L1-cuboid crowding mis-reading curved 3-objective fronts, so DTLZ2 HV may land below RVEA's — the exact hole
the reference-point and indicator rungs above it close.

**Hyperparameters.** SBX `eta_c = 20` at crossover probability **0.9** (note: lower than RVEA's 1.0),
polynomial mutation `eta_m = 20`, `p_m = 1/n_var`; binary DCD tournament for mating; rank-then-crowding
truncation for survival. (`selTournamentDCD` needs `pop_size` divisible by 4; the task sizes 100, 120 satisfy
this.)

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 4: NSGA-II
class CustomMOEA:
    """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        """Binary tournament selection with crowding distance."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

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
        """NSGA-II survival: non-dominated sorting + crowding distance."""
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break

        return next_gen

    def on_generation(self, gen, population):
        pass
```
