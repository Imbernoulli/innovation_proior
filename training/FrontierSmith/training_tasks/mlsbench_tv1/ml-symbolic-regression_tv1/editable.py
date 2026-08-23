def fitness_function(tree, X, y, complexity_weight=0.0):
    """Fitness of a candidate program (lower is better).

    Variant objective: fitness should be a proxy for TRANSFER, not just
    training fit. ``complexity_weight`` scales a per-node size penalty; at
    the default 0.0 the fitness is pure training MSE and offers no
    protection against memorizing trees whose fit will not survive the
    held-out test inputs. Signals worth adding here: size/depth pressure,
    agreement across subsamples of (X, y), penalties for near-singular
    protected-operator behavior.
    """
    y_pred = safe_evaluate(tree, X)
    mse = float(np.mean((y - y_pred) ** 2))
    return mse + complexity_weight * tree.size()


def selection(population, fitnesses, n_select, tournament_size=7):
    """Select ``n_select`` individuals (copies) for reproduction.

    Placeholder: uniform random choice that ignores both fitness and
    ``tournament_size`` — no exploitation pressure at all. Replace with a
    scheme that balances exploitation against the behavioral diversity
    the stagnation tracker in ``evolve_one_generation`` depends on.
    """
    selected = []
    for _ in range(n_select):
        idx = random.randint(0, len(population) - 1)
        selected.append(population[idx].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Produce one offspring from two parents.

    Placeholder: returns an unchanged copy of ``parent1`` — no
    recombination, so building blocks never mix and stalls are permanent.
    """
    return parent1.copy()


def mutation(parent, n_features, max_depth=17):
    """Produce a mutated copy of ``parent``.

    Placeholder: no-op copy — the search cannot inject novelty, which is
    exactly what the stall counter below needs to trigger.
    """
    return parent.copy()


# Cross-generation reliability state: tracks whether the search is still
# making progress. An improved strategy should ACT on stall_generations
# (restarts from the elite, boosted mutation, enforced novelty) instead of
# merely recording it like the placeholder does.
_SEARCH_STATE = {"best_fitness": float("inf"), "stall_generations": 0}


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation from the current population.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        X_train, y_train: training samples (the ONLY target information)
        n_features: number of input features
        pop_size: desired population size
        crossover_rate, mutation_rate: operator probabilities
        max_depth: maximum allowed tree depth

    Returns:
        list of Node — next generation population (length pop_size)

    ``_SEARCH_STATE['stall_generations']`` counts generations without an
    improvement of the best training fitness. The placeholder records the
    statistic but never reacts, so premature convergence goes uncorrected
    — the reliability failure this variant targets.
    """
    best = float(np.min(fitnesses))
    if best < _SEARCH_STATE["best_fitness"] - 1e-12:
        _SEARCH_STATE["best_fitness"] = best
        _SEARCH_STATE["stall_generations"] = 0
    else:
        _SEARCH_STATE["stall_generations"] += 1

    new_population = []
    # Elitism: keep the best individual
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        parents = selection(population, fitnesses, 2)
        r = random.random()
        if r < crossover_rate:
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
