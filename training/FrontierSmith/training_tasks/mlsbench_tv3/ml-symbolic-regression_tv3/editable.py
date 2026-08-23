def fitness_function(tree, X, y, trim_frac=0.1):
    """Trimmed squared-error fitness (lower is better).

    Variant objective: never let the largest residuals dictate the score.
    The worst ``trim_frac`` of squared residuals is dropped before
    averaging, so a candidate is judged on how it explains the bulk of the
    sample rather than on a handful of extreme points — the points most
    likely to reflect sampling artifacts or protected-operator edge
    behavior. Stronger robust losses (Huber, median-of-means) and
    subsample-consensus checks are the intended upgrades.
    """
    y_pred = safe_evaluate(tree, X)
    sq = (y - y_pred) ** 2
    keep = max(1, int(round(sq.shape[0] * (1.0 - trim_frac))))
    return float(np.mean(np.sort(sq)[:keep]))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Sequential-scan tournament on the robust fitness.

    Draws ``tournament_size`` random contenders one at a time and keeps
    the best under the trimmed loss. No size tiebreak and no diversity
    bonus in the scaffold — the robustness story lives in the loss and in
    the gentleness of the variation operators.
    """
    chosen = []
    n = len(population)
    for _ in range(n_select):
        best_i = random.randint(0, n - 1)
        for _ in range(tournament_size - 1):
            j = random.randint(0, n - 1)
            if fitnesses[j] < fitnesses[best_i]:
                best_i = j
        chosen.append(population[best_i].copy())
    return chosen


def crossover(parent1, parent2, n_features, max_depth=17):
    """Depth-guarded subtree exchange between two parents.

    A random donor subtree from ``parent2`` overwrites a random site in a
    copy of ``parent1``; offspring that exceed the depth cap revert to
    ``parent1``. Structural churn is allowed here — the noise-robustness
    burden is carried by the trimmed loss deciding which offspring live.
    """
    offspring = parent1.copy()
    sites = offspring.get_all_nodes()
    _, site_parent, site_idx = sites[random.randint(0, len(sites) - 1)]
    donor_pool = parent2.get_all_nodes()
    donor = donor_pool[random.randint(0, len(donor_pool) - 1)][0].copy()
    if site_parent is None:
        offspring = donor
    else:
        site_parent.children[site_idx] = donor
    if offspring.depth() > max_depth:
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    """Point mutation: measured local edits, never wholesale rewrites.

    One random node is edited in place. Operators swap to a same-arity
    peer, constants get a small multiplicative/additive nudge, variables
    are re-drawn. The tree's shape is preserved, so mutation explores the
    neighborhood of a structure instead of trading it for a new one that
    might fit stray points by luck.
    """
    child = parent.copy()
    target = child.get_all_nodes()[random.randint(0, child.size() - 1)][0]
    if target.is_terminal:
        if str(target.value).startswith('x'):
            target.value = f'x{random.randint(0, n_features - 1)}'
        else:
            val = float(target.value)
            target.value = str(round(val * random.uniform(0.8, 1.2)
                                     + random.uniform(-0.1, 0.1), 4))
    else:
        arity = OPERATORS[target.value][1]
        peers = [op for op in OPERATOR_NAMES
                 if OPERATORS[op][1] == arity and op != target.value]
        if peers:
            target.value = random.choice(peers)
    return child


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation under the robust (trimmed) loss.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        X_train, y_train: training samples (possibly noisy witnesses)
        n_features: number of input features
        pop_size: desired population size
        crossover_rate, mutation_rate: operator probabilities
        max_depth: maximum allowed tree depth

    Returns:
        list of Node — next generation population (length pop_size)

    The elite is the best tree under the trimmed loss. A natural upgrade
    is consensus elitism: re-score the top few candidates on random
    subsamples of (X_train, y_train) and promote the one whose rank is
    stable across subsamples, not the one with the single best score.
    """
    elite = population[int(np.argmin(fitnesses))].copy()
    children = [elite]
    for _ in range(pop_size - 1):
        p1, p2 = selection(population, fitnesses, 2)
        roll = random.random()
        if roll < crossover_rate:
            children.append(crossover(p1, p2, n_features, max_depth))
        elif roll < crossover_rate + mutation_rate:
            children.append(mutation(p1, n_features, max_depth))
        else:
            children.append(p1)
    return children[:pop_size]
