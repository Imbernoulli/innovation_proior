
# ----------------------------------------------------------------
# Variant re-aim: RESTART CALENDAR + POPULATION-SIZING POLICY. The
# operators are deliberately ordinary (tournament, two-point
# crossover, polynomial mutation); the contribution surface is the
# policy deciding when a trajectory is abandoned, how large the next
# epoch's population is, and what crosses the boundary. One policy
# serves all four settings; every epoch, re-seeding included, draws
# from one shared evaluation pool.
# ----------------------------------------------------------------

# Policy constants — the intended redesign surface.
STALL_LIMIT = 60      # generations without a new archived best => abandon
POP_GROWTH = 1.5      # multiplier on the next epoch's population
POP_CAP_FACTOR = 3    # ceiling on epoch population, in units of pop_size


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Select k individuals from the population.

    Placeholder parent selection: tournament of size 3. Ordinary by
    design — this variant's contribution lives in the restart policy,
    not here.
    """
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Apply crossover to two individuals (modified in place).

    Placeholder recombination: two-point crossover. Genes are copied
    between parents, so offspring remain inside the domain.
    """
    tools.cxTwoPoint(ind1, ind2)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Apply mutation to an individual.

    Placeholder: bounded polynomial mutation, eta=25, two expected genes
    changed per individual.
    """
    tools.mutPolynomialBounded(
        individual, eta=25.0, low=lo, up=hi,
        indpb=min(1.0, 2.0 / len(individual)),
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
    """Multi-epoch loop: evolve, detect stagnation, restart re-sized.

    The global best is archived across epochs and is what the run
    returns; fitness_history records the archived best per generation,
    so a restart never erases progress from the record. All epochs
    (including the cost of re-seeding a fresh population) are charged
    to the single harness-implied evaluation pool, and the loop stops
    when it cannot afford the next payment.

    Returns:
        best_individual: the best individual found across all epochs.
        fitness_history: list of archived best fitness per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Shared evaluation pool across every epoch
    eval_cap = pop_size * (n_generations + 1)
    evals_used = 0

    # Epoch 1 population
    epoch_pop = pop_size
    pop = toolbox.population(n=epoch_pop)
    evals_used += len(pop)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    archive = toolbox.clone(min(pop, key=lambda ind: ind.fitness.values[0]))
    stall = 0
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

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        # Clip to bounds
        for ind in offspring:
            clip_individual(ind, lo, hi)

        # Evaluate individuals with invalid fitness (pool-charged)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if evals_used + len(invalid_ind) > eval_cap:
            break
        evals_used += len(invalid_ind)
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Archive across epochs; stagnation bookkeeping
        gen_best = min(pop, key=lambda ind: ind.fitness.values[0])
        if gen_best.fitness.values[0] < archive.fitness.values[0]:
            archive = toolbox.clone(gen_best)
            stall = 0
        else:
            stall += 1

        # Track the archived best (restarts never falsify history)
        best_fit = archive.fitness.values[0]
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

        # Restart policy: abandon the epoch, re-size, re-seed
        if stall >= STALL_LIMIT:
            epoch_pop = min(
                int(epoch_pop * POP_GROWTH), POP_CAP_FACTOR * pop_size
            )
            if evals_used + epoch_pop > eval_cap:
                break
            pop = [toolbox.individual() for _ in range(epoch_pop)]
            evals_used += len(pop)
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            stall = 0

    best_ind = archive
    return best_ind, fitness_history
