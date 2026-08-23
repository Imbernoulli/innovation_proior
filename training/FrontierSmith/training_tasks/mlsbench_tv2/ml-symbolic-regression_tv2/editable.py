HARD_SIZE_BUDGET = 24  # node ceiling: trees above this are ineligible, not "worse"


def fitness_function(tree, X, y, size_budget=HARD_SIZE_BUDGET):
    """Budget-first fitness (lower is better).

    Variant objective: the node budget is a HARD constraint. Trees at or
    under ``size_budget`` compete on plain training MSE; trees over it are
    pushed behind the admissible set by a wall penalty that grows with the
    overshoot. There is deliberately NO soft size term blended into the
    loss of admissible trees — compactness is enforced, not traded.
    """
    y_pred = safe_evaluate(tree, X)
    mse = float(np.mean((y - y_pred) ** 2))
    overshoot = tree.size() - size_budget
    if overshoot > 0:
        return mse + 1e6 * overshoot
    return mse


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection with a deterministic small-tree tiebreak.

    Each pick runs a size-``tournament_size`` tournament ordered by
    ``(fitness, tree size)`` — equal fits resolve toward the smaller tree,
    so the budget keeps biting even when errors saturate.
    """
    selected = []
    n = len(population)
    for _ in range(n_select):
        contenders = [random.randint(0, n - 1) for _ in range(tournament_size)]
        winner = min(contenders, key=lambda i: (fitnesses[i], population[i].size()))
        selected.append(population[winner].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17,
              size_budget=HARD_SIZE_BUDGET):
    """Budget-guarded subtree graft.

    A random subtree of ``parent2`` replaces a random site in a copy of
    ``parent1``. The graft is accepted only if the offspring respects both
    the depth cap and the node budget; otherwise the smaller parent is
    returned unchanged, so recombination can never inflate past the cap.
    """
    child = parent1.copy()
    sites = child.get_all_nodes()
    _, site_parent, site_idx = sites[random.randint(0, len(sites) - 1)]
    donors = parent2.get_all_nodes()
    graft = donors[random.randint(0, len(donors) - 1)][0].copy()
    if site_parent is None:
        child = graft
    else:
        site_parent.children[site_idx] = graft
    if child.size() > size_budget or child.depth() > max_depth:
        smaller = parent1 if parent1.size() <= parent2.size() else parent2
        return smaller.copy()
    return child


def mutation(parent, n_features, max_depth=17):
    """Shrink mutation: replace a random internal subtree with a terminal.

    The only mutation in the scaffold is deflationary — it deletes
    structure. Growth happens solely through guarded crossover, keeping
    the population's size distribution pressed against the budget from
    below. (Hoist mutation and other shrink moves are natural extensions.)
    """
    child = parent.copy()
    internal = [(nd, par, i) for nd, par, i in child.get_all_nodes()
                if par is not None and not nd.is_terminal]
    if not internal:
        return child
    _, par, i = internal[random.randint(0, len(internal) - 1)]
    par.children[i] = random_terminal(n_features)
    return child


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation under the hard size budget.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        X_train, y_train: training samples (the only target information)
        n_features: number of input features
        pop_size: desired population size
        crossover_rate, mutation_rate: operator probabilities
        max_depth: maximum allowed tree depth

    Returns:
        list of Node — next generation population (length pop_size)

    Elitism also respects the tiebreak: among equally fit candidates the
    smallest tree is carried forward, so the incumbent is always the most
    parsimonious expression seen at the current error level.
    """
    order = sorted(range(len(population)),
                   key=lambda i: (fitnesses[i], population[i].size()))
    next_gen = [population[order[0]].copy()]

    while len(next_gen) < pop_size:
        parents = selection(population, fitnesses, 2)
        r = random.random()
        if r < crossover_rate:
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = parents[0]
        next_gen.append(child)

    return next_gen[:pop_size]
