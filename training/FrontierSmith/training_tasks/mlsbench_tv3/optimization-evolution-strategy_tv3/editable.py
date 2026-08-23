
# ----------------------------------------------------------------
# Variant re-aim: DIMENSION-ROBUST SCALING. The same code and the same
# constants must hold at n=30 and n=100; every variation quantity
# carries an explicit dependence on dim. A diagonal per-coordinate
# scale model (_CoordScale, O(n) memory and time) drives mutation;
# survivor selection is (mu+lambda) truncation, whose pressure does
# not dilute as dimension grows. The rastrigin-30d vs rastrigin-100d
# gap is the quantity under test.
# ----------------------------------------------------------------


class _CoordScale:
    """Per-coordinate spread model with a dimension-scaled floor.

    sigma[i] tracks the surviving population's standard deviation along
    coordinate i (EMA, weight 0.3); mutation draws at this scale, so step
    lengths follow the population geometry instead of a constant tuned at
    one dimensionality. The floor, proportional to span/sqrt(dim),
    prevents collapse. Everything here is O(n) per generation — the
    deliberate structural budget of this variant.
    """

    def __init__(self):
        self.sigma = None
        self.floor = 0.0

    def reset(self, dim: int, lo: float, hi: float) -> None:
        span = hi - lo
        self.floor = 1e-3 * span / math.sqrt(dim)
        self.sigma = np.full(dim, 0.3 * span / math.sqrt(dim))

    def update(self, population) -> None:
        arr = np.asarray(population, dtype=float)
        self.sigma = np.maximum(
            0.7 * self.sigma + 0.3 * arr.std(axis=0), self.floor
        )


_SCALE = _CoordScale()


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Select k parents.

    Placeholder: tournament of size 3. In this design the survivor step
    inside run_evolution, not this parent step, carries the selection
    pressure, so its intensity stays dimension-independent.
    """
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Apply crossover to two individuals (modified in place).

    Uniform per-coordinate exchange (indpb=0.5): acts coordinate-wise and
    therefore keeps the same character at every dimension, unlike blend
    or segment operators whose effect drifts with n.
    """
    tools.cxUniform(ind1, ind2, indpb=0.5)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Apply mutation at the per-coordinate model scale.

    The expected number of mutated genes is held at one regardless of
    dimension (indpb=1/dim), and each perturbation is drawn at
    _SCALE.sigma[i], so total step length follows the population spread
    rather than a hand-tuned constant.
    """
    n = len(individual)
    indpb = 1.0 / n
    for i in range(n):
        if random.random() < indpb:
            individual[i] += random.gauss(0.0, float(_SCALE.sigma[i]))
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
    """(mu+lambda) loop with an O(n) per-coordinate scale model.

    Variant objective: identical behaviour, identical constants, at 30
    and at 100 dimensions. Survivors are the best pop_size of parents
    plus offspring (truncation — pressure independent of dim); the scale
    model is refreshed from the survivors each generation.

    Returns:
        best_individual: the best individual found.
        fitness_history: list of best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    _SCALE.reset(dim, lo, hi)

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

    fitness_history = []

    for gen in range(n_generations):
        # Parent selection
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover (coordinate-wise)
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation (model-scaled)
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

        # Survivor truncation over parents + offspring: (mu+lambda)
        combined = pop + offspring
        combined.sort(key=lambda ind: ind.fitness.values[0])
        pop[:] = combined[:pop_size]

        # Refresh the per-coordinate scale model from the survivors
        _SCALE.update(pop)

        # Track best fitness
        best_fit = pop[0].fitness.values[0]
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
