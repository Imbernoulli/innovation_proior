
# ----------------------------------------------------------------
# Variant re-aim: WORST-LANDSCAPE floor under ONE fixed, restart-free
# configuration. The four benchmark settings are combined by a geometric
# mean, so the score is governed by the weakest landscape. Adaptation must
# come from statistics the run measures about itself (see _AdaptState),
# consumed as continuous quantities — never per-function/per-dimension
# branching, never restarts. The evaluation budget (initial population +
# at most pop_size offspring per generation) is tracked explicitly.
# ----------------------------------------------------------------


class _EvalBudget:
    """Explicit fitness-evaluation accountant (restart-free, hard-capped).

    The harness fixes the budget implicitly through pop_size and
    n_generations; this object makes it explicit so a redesigned loop
    (steady-state, variable offspring counts, ...) cannot silently spend
    more evaluations than the standard generational loop would.
    """

    def __init__(self, pop_size: int, n_generations: int):
        self.cap = pop_size * (n_generations + 1)
        self.used = 0

    def spend(self, n: int) -> bool:
        """Reserve n evaluations; returns False if the cap would be exceeded."""
        if self.used + n > self.cap:
            return False
        self.used += n
        return True


class _AdaptState:
    """Continuous self-measurement of the run (the adaptation channel).

    Tracks the offspring success rate (fraction of newly evaluated
    offspring improving on the previous generation's median fitness) and
    the population spread. Both are statistics of the run itself, not of
    the benchmark identity, and are the only admissible inputs for
    adapting operator intensity in this variant.
    """

    def __init__(self):
        self.success_rate = 0.2
        self.spread = None

    def update(self, prev_median: float, offspring_fits: list, population) -> None:
        if offspring_fits:
            n_better = sum(1 for f in offspring_fits if f < prev_median)
            inst = n_better / len(offspring_fits)
            self.success_rate = 0.9 * self.success_rate + 0.1 * inst
        arr = np.asarray(population, dtype=float)
        self.spread = float(np.mean(np.std(arr, axis=0)))

    def mutation_eta(self) -> float:
        """Map the success rate to a polynomial-mutation index, continuously.

        Low success => broaden mutation (smaller eta); high success =>
        refine locally (larger eta). The placeholder mapping is
        deliberately mild — this hook is the intended redesign surface.
        """
        return float(np.clip(10.0 + 60.0 * self.success_rate, 10.0, 40.0))


_ADAPT = _AdaptState()


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Select k individuals from the population.

    Placeholder: tournament selection, size 3 (same contract as the base
    interface: individuals expose .fitness.values, minimisation).
    """
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Apply crossover to two individuals (modified in place).

    Placeholder: simulated binary crossover with a fixed eta=20 — one
    setting for every landscape, per the variant's constraint.
    """
    tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Apply mutation to an individual.

    The mutation index follows the run's own success-rate statistic
    (continuous adaptation channel); per-gene probability scales with
    dimension so 30d and 100d see comparable per-individual change.
    Genes are clipped into bounds first: polynomial mutation is only
    well-defined for in-bounds genes (a gene left outside the domain by
    unbounded crossover makes its bounded perturbation ill-posed).
    """
    clip_individual(individual, lo, hi)
    tools.mutPolynomialBounded(
        individual, eta=_ADAPT.mutation_eta(), low=lo, up=hi,
        indpb=1.0 / len(individual),
    )
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
    """Run one restart-free evolutionary trajectory under a hard eval cap.

    Variant objective: make the WEAKEST of the four benchmark settings as
    strong as possible with this single configuration. The loop keeps an
    explicit evaluation ledger (_EvalBudget) and feeds the adaptation
    state (_AdaptState) once per generation; premature convergence must
    be handled by the operators — the loop never re-initialises.

    Returns:
        best_individual: the best individual found.
        fitness_history: list of best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Fresh self-measurement for this trajectory
    _ADAPT.__init__()
    budget = _EvalBudget(pop_size, n_generations)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population (counts against the ledger)
    pop = toolbox.population(n=pop_size)
    budget.spend(len(pop))
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(n_generations):
        prev_median = float(np.median([ind.fitness.values[0] for ind in pop]))

        # Selection
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        # Clip crossover products into the domain, then mutate (bounded
        # mutation assumes in-bounds genes), then clip again.
        for ind in offspring:
            clip_individual(ind, lo, hi)
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values
        for ind in offspring:
            clip_individual(ind, lo, hi)

        # Evaluate individuals with invalid fitness — through the ledger
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if not budget.spend(len(invalid_ind)):
            break  # hard cap: never exceed the harness-implied budget
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population (single trajectory — no restarts)
        pop[:] = offspring

        # Feed the continuous adaptation channel
        _ADAPT.update(prev_median, [ind.fitness.values[0] for ind in invalid_ind], pop)

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
