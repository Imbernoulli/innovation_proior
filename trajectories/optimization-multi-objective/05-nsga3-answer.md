**Problem.** NSGA-II is stable and parameter-free and wins the 2-objective problems, but its crowding distance
is an L1-cuboid density estimate: on a curved 3-objective front, per-axis neighbour gaps stop measuring real
crowding, so DTLZ2 came in with the worst spread of any rung (0.653) and HV below RVEA (2.7407). The fix must
keep NSGA-II's knob-free elitist framework and replace only the high-dimensional split.

**Key idea.** Stop *estimating* diversity from a sparse cloud; *impose* it from a fixed grid of evenly spaced
reference directions (Das–Dennis simplex lattice). Keep the combined pool, fast sort, and fill-fronts; replace
the last-front split with reference-point niching: normalize the front each generation (ideal + ASF extreme
points + hyperplane intercepts), associate each member to the reference *line* of minimum perpendicular
distance (which decouples convergence = position-along-ray from diversity = which-ray), and fill the
emptiest reference direction first. Perpendicular-to-line association reads a curved front where the cuboid
could not.

**Why it ranks above NSGA-II.** It keeps everything NSGA-II got right and repairs the one thing it got wrong,
so it should match the ZDTs and beat NSGA-II on DTLZ2 HV and spread. Note this harness delegates `survive`
entirely to DEAP's `selNSGA3` (full normalization + niching), but unlike the canonical many-objective form it
leaves `cx_eta = 20` (not 30) and applies crossover always — slightly more exploratory variation.

**Hyperparameters.** Reference points `p = pop_size-1` (2-obj) / `p = 12` (3-obj); SBX `eta_c = 20` applied at
probability 1.0; polynomial mutation `eta_m = 20`, `p_m = 1/n_var`; random-shuffle mating (diversity is a
survival property). No convergence-vs-diversity knob.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 5: NSGA-III
class CustomMOEA:
    """NSGA-III: Non-dominated Sorting Genetic Algorithm III."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

        # Generate reference points
        if n_obj == 2:
            p = pop_size - 1  # number of divisions
            self.ref_points = tools.uniform_reference_points(n_obj, p=p)
        else:
            self.ref_points = tools.uniform_reference_points(n_obj, p=12)

    def select(self, population, k):
        """Random shuffle selection (NSGA-III relies on survive for diversity)."""
        selected = [deepcopy(ind) for ind in population]
        random.shuffle(selected)
        return selected[:k]

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 1.0:
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
        """NSGA-III survival: reference-point-based selection."""
        combined = population + offspring

        # Use DEAP's built-in NSGA-III selection
        selected = tools.selNSGA3(combined, self.pop_size, self.ref_points)
        return selected

    def on_generation(self, gen, population):
        pass
```
