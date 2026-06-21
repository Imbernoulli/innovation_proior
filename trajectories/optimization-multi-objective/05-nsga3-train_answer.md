NSGA-II delivered the stability I climbed to it for and in doing so drew the boundary of what crowding distance can do. On the 2-objective ZDTs it is the cleanest yet (ZDT1 HV 0.8682, spread 0.343, all three seeds tight to within 0.001 on HV), and on DTLZ1 the blow-up seed is gone (spread mean 0.904, worst seed 1.49). But the pivotal prediction held: on DTLZ2 its HV is 2.7407 — *below* RVEA's 2.7886 — with spread 0.653, the worst of the four rungs on that problem. That is the $L_1$-cuboid crowding distance mis-reading the curved sphere. Crowding distance sorts the critical front by each objective in turn and sums each point's normalized gaps to its two flanking neighbors along that axis; in 2D the neighbor along $f_1$ genuinely *is* the neighbor on the front, but on three axes "nearest neighbor along axis 3" has little to do with "nearest neighbor in the full 3-vector," so the sum-of-axiswise-gaps stops measuring real crowding and the split degenerates toward random. The fix is not to re-introduce a knob — that was RVEA's mistake — but to keep NSGA-II's entire knob-free elitist framework and replace *only* the diversity device.

I propose **NSGA-III**: reference-point niching. Stop *estimating* diversity from the sparse point cloud and instead *impose* it from a fixed external set of target directions, decided ahead of time and independent of the population, so it cannot flatten no matter the geometry. Where should the targets be? Das and Dennis's simplex-lattice gives a systematic, scale-independent way to lay evenly spaced points on the unit simplex (the hyperplane $\sum_i w_i = 1$, $w_i \ge 0$): with $p$ divisions, enumerate every vector whose components are in $\{0/p, \dots, p/p\}$ and sum to one. These tile the simplex with equal nearest-neighbor spacing and do not depend on the front's shape, because they live on the abstract simplex, not on the front. The harness builds them with `tools.uniform_reference_points`: $p = \text{pop\_size}-1$ for two objectives (roughly one reference line per population slot, a dense even fan), $p = 12$ for three. Each lattice point defines a reference *line*, a ray from the origin; the plan keeps NSGA-II's framework and replaces only the last-front split so that survivors are handed to reference directions, every direction getting a representative — diversity supplied by the grid and enforced by the fill, never estimated from the cloud.

Two pieces make the split work, both inside DEAP's `selNSGA3`, which the harness delegates to wholesale. First, **normalization**, redone every generation because the objectives sit on shifting scales while the grid lives on a clean 0-to-1 simplex. Translate so the per-axis ideal (minimum over the kept fronts) is the origin; find the $M$ extreme points — *not* by raw argmax, which picks dominated outliers, but by minimizing the achievement scalarizing function $\max_i f'_i / w^j_i$ with $w^j$ heavy on axis $j$ and tiny ($10^{-6}$) off it, so it finds the point squarely along each axis; solve the one $M\times M$ system for the hyperplane through those extremes and read its axis intercepts as the per-axis scale, with robust fallbacks to the worst values when the solve degenerates. This lands the front roughly on the same simplex as the grid, re-fitted to wherever the population currently is — which is what lets it handle the differently-scaled DTLZ fronts with no tuning. Second, **association by perpendicular distance to the reference line**, not to the reference point. A reference point defines a *direction*, and a solution serves that direction regardless of how far out along the ray it sits — position along the ray is convergence (the rank's job), perpendicular offset is which-direction-am-I (the split's job). This is the decoupling crowding never had, and it is exactly why it reads a curved front correctly: the geometry of the sphere does not corrupt a perpendicular-to-line measurement the way it corrupts a sum-of-axis-gaps. The split then fills the *emptiest* reference direction first — count how many already-kept members associate to each reference point, repeatedly serve the least-represented one, seeding a brand-new niche with its closest member and topping up an existing niche with any member — until the population is full. Nothing estimates the population's own density, so nothing degrades on the sphere.

Two task-specific divergences matter for the numbers. The canonical many-objective form pushes the SBX distribution index *up* to $\eta_c = 30$ — keeping children near their parents to counter the failure mode where two distant parents in a spread-out population produce an offspring in an unexplored void. This edit surface does *not*: it leaves $\eta_c = 20$, the scaffold default, and applies crossover with probability **1.0** (always), unlike NSGA-II's 0.9. So variation is slightly more exploratory than the canonical NSGA-III — a difference I note because on the 3-objective problems the un-raised $\eta_c$ could leave a touch more variance, though with only three objectives here it should be minor. Mating selection is a plain random shuffle: diversity is now entirely a survival property, so there is no need to bias mating, and there is no fragile diversity knob anywhere — the only structural choice is the reference grid resolution, which is a coverage knob, not a convergence-vs-diversity tradeoff. The whole novelty is in `survive`, the single call `tools.selNSGA3(combined, pop_size, ref_points)`.

I expect NSGA-III to *match or slightly beat* NSGA-II on the ZDTs — with $p = \text{pop\_size}-1$ reference lines the grid is dense enough to spread a 2-objective front at least as evenly as crowding, and because the grid imposes spread from outside, ZDT1 spread may drop noticeably below NSGA-II's 0.343 — and to *beat* NSGA-II on DTLZ2 on both HV (above 2.7407, plausibly past RVEA's 2.7886) and spread, because the perpendicular-to-line association reads the sphere where the cuboid could not. On DTLZ1 I expect stability comparable to NSGA-II. That crossover — match the ZDTs, win DTLZ2 — is the move the ranking needs, and it sets up the final question: NSGA-III still optimizes coverage of a *fixed reference grid*, which is only a proxy for the actual point-cloud density the Spread metric scores, so the rung above must measure diversity against the front's real density rather than against a pre-laid grid.

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
