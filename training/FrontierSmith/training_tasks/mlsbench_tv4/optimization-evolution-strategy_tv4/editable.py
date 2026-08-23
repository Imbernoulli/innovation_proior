
# ----------------------------------------------------------------
# Variant re-aim: RANK-ONLY FITNESS DISCIPLINE. Fitness values are
# compared, never computed with: selection weights come from rank
# positions, step-size adaptation from a success count against an
# order statistic (Rechenberg's one-fifth rule), and the returned
# archive is kept by comparison. The trajectory is invariant to any
# strictly increasing transformation of the objective; the claim is
# that on this suite the discipline costs nothing.
# ----------------------------------------------------------------


class _FifthRule:
    """Global step-size state driven purely by comparisons.

    frac is the mutation scale as a fraction of the domain span. Each
    generation the loop reports the fraction of newly evaluated
    offspring that beat the parent generation's lower-median (an order
    statistic, deliberately not an average); the classic multiplicative
    update follows — expand above one-fifth success, contract below.
    No fitness magnitude ever enters this object.
    """

    def __init__(self):
        self.frac = 0.15

    def update(self, success_rate: float) -> None:
        if success_rate > 0.2:
            self.frac = min(self.frac * 1.22, 0.5)
        else:
            self.frac = max(self.frac / 1.22, 1e-4)


_STEP = _FifthRule()


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Linear-ranking selection.

    Sort by fitness (comparisons only), weight each individual by its
    rank position, and sample k with replacement. Fitness magnitudes
    never become weights — only positions do.
    """
    ranked = sorted(population, key=lambda ind: ind.fitness.values[0])
    n = len(ranked)
    weights = [n - i for i in range(n)]
    return random.choices(ranked, weights=weights, k=k)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Intermediate recombination with a fresh convex weight per gene.

    A pure genotype operation: children stay inside their parents' box,
    so no bound handling is induced and no fitness is touched.
    """
    for i in range(len(ind1)):
        a = random.random()
        x, y = ind1[i], ind2[i]
        ind1[i] = a * x + (1.0 - a) * y
        ind2[i] = (1.0 - a) * x + a * y
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Gaussian mutation at the one-fifth-rule scale.

    Sigma is _STEP.frac times the domain span; one gene is changed per
    individual in expectation. The scale is set entirely by comparison
    outcomes accumulated in _STEP.
    """
    sigma = _STEP.frac * (hi - lo)
    indpb = 1.0 / len(individual)
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] += random.gauss(0.0, sigma)
    return (individual,)


def run_evolution(
    evaluate_func: Callable,
    dim: int,
    lo: float,
    hi: float,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Tuple[list, list]:
    """Comparison-driven generational loop.

    Variant discipline: fitness values are consumed only through
    comparisons and order statistics — ranking for selection, a success
    count against the parents' lower-median for step-size adaptation,
    and a compared-and-kept archive for the returned best. The
    trajectory is therefore unchanged by any strictly increasing
    rescaling of the objective.

    Returns:
        best_individual: the best individual found (archive).
        fitness_history: list of best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    _STEP.__init__()

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    archive = toolbox.clone(min(pop, key=lambda ind: ind.fitness.values[0]))
    fitness_history = []

    for gen in range(n_generations):
        # Lower-median of the parents: an order statistic, not np.median
        fits = sorted(ind.fitness.values[0] for ind in pop)
        threshold = fits[(len(fits) - 1) // 2]

        # Selection (rank-weighted)
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation (one-fifth-rule scale)
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        # Clip to bounds
        for ind in offspring:
            clip_individual(ind, lo, hi)

        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Success = beat the parents' lower-median (a comparison)
        n_better = sum(
            1 for ind in invalid_ind if ind.fitness.values[0] < threshold
        )
        _STEP.update(n_better / max(1, len(invalid_ind)))

        # Replace population
        pop[:] = offspring

        # Archive kept by comparison
        gen_best = min(pop, key=lambda ind: ind.fitness.values[0])
        if gen_best.fitness.values[0] < archive.fitness.values[0]:
            archive = toolbox.clone(gen_best)

        # Track best fitness
        best_fit = gen_best.fitness.values[0]
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    best_ind = archive
    return best_ind, fitness_history
