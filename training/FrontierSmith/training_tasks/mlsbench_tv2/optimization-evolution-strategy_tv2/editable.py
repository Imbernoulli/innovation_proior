
# ----------------------------------------------------------------
# Variant re-aim: EVALUATION-BUDGET EFFICIENCY. The run is judged on
# fitness gained per evaluation spent: best_fitness must be reached
# with a small convergence_gen, on all four settings. The loop keeps
# a spend ledger (_SpendState), pays only for genomes that variation
# actually changed, and anneals variation intensity against the
# remaining allowance so improvement is front-loaded by construction.
# ----------------------------------------------------------------


class _SpendState:
    """Evaluation-spend accounting plus the annealing signal.

    cap is the harness-implied allowance (initial population plus up to
    pop_size offspring per generation). frac_left in [0, 1] is exposed to
    the variation operators as the annealing clock: exploration while the
    pool is full, refinement as it drains. This object is the intended
    redesign surface for smarter spend schedules.
    """

    def __init__(self):
        self.cap = 0
        self.used = 0
        self.frac_left = 1.0

    def reset(self, pop_size: int, n_generations: int) -> None:
        self.cap = pop_size * (n_generations + 1)
        self.used = 0
        self.frac_left = 1.0

    def afford(self, n: int) -> bool:
        """True iff n more evaluations fit inside the allowance."""
        return self.used + n <= self.cap

    def charge(self, n: int) -> None:
        self.used += n
        if self.cap > 0:
            self.frac_left = max(0.0, 1.0 - self.used / self.cap)


_SPEND = _SpendState()


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Select k individuals from the population.

    Placeholder: tournament selection with tournament size 4 — pressure
    turned up slightly so that paid-for improvements propagate quickly.
    """
    return tools.selTournament(population, k, tournsize=4)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Apply crossover to two individuals (modified in place).

    Placeholder: blend crossover (BLX), alpha=0.3 — children sample the
    segment around their parents, reusing information that has already
    been paid for. The loop clips genes the blend pushes out of bounds.
    """
    tools.cxBlend(ind1, ind2, alpha=0.3)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Apply mutation to an individual.

    Gaussian perturbation whose scale follows the spend clock: sigma
    shrinks linearly from 20% to 2% of the domain span as the allowance
    drains, so early evaluations buy exploration and late ones buy
    refinement. Expected genes changed per individual: one.
    """
    sigma = (0.02 + 0.18 * _SPEND.frac_left) * (hi - lo)
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
    """Run the evolutionary loop under an explicit spend ledger.

    Variant objective: the largest fitness gain per evaluation spent.
    Only genomes changed by variation are (re)evaluated, one elite is
    carried between generations so paid-for progress is never lost, and
    the ledger hard-stops the loop before the allowance is overdrawn.

    Returns:
        best_individual: the best individual found.
        fitness_history: list of best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    _SPEND.reset(pop_size, n_generations)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population (charged to the ledger)
    pop = toolbox.population(n=pop_size)
    _SPEND.charge(len(pop))
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    elite = toolbox.clone(min(pop, key=lambda ind: ind.fitness.values[0]))
    fitness_history = []

    for gen in range(n_generations):
        # Selection
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Mutation (annealed by the spend clock)
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        # Clip to bounds
        for ind in offspring:
            clip_individual(ind, lo, hi)

        # Pay only for changed genomes; stop before overdrawing the pool.
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if not _SPEND.afford(len(invalid_ind)):
            break
        _SPEND.charge(len(invalid_ind))
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Elite carry: paid-for progress is never lost to variation.
        worst_i = max(range(len(pop)), key=lambda i: pop[i].fitness.values[0])
        if elite.fitness.values[0] < pop[worst_i].fitness.values[0]:
            pop[worst_i] = toolbox.clone(elite)
        gen_best = min(pop, key=lambda ind: ind.fitness.values[0])
        if gen_best.fitness.values[0] < elite.fitness.values[0]:
            elite = toolbox.clone(gen_best)

        # Track best fitness
        best_fit = min(ind.fitness.values[0] for ind in pop)
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
