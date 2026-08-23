def _numeric_leaves(tree):
    """List the constant-valued terminal Nodes of *tree* (live references)."""
    return [nd for nd, _, _ in tree.get_all_nodes()
            if nd.is_terminal and not str(nd.value).startswith('x')]


def tune_constants(tree, X, y, n_probes=3):
    """Cheap coefficient refinement: jitter all constants, keep improvements.

    The structure/coefficient split of this variant lives here. Each probe
    perturbs every numeric leaf of the incumbent with Gaussian noise scaled
    to the leaf's magnitude and keeps the probe only if training MSE drops.
    Deliberately crude — ``n_probes`` extra evaluations per call — so the
    evaluation budget stays honest. Coordinate descent, golden-section
    steps, or linear-scaling tricks are the intended replacements.
    """
    best = tree.copy()
    best_fit = fitness_function(best, X, y)
    for _ in range(n_probes):
        trial = best.copy()
        leaves = _numeric_leaves(trial)
        if not leaves:
            break
        for leaf in leaves:
            val = float(leaf.value)
            leaf.value = str(round(val + random.gauss(0.0, 0.3 + 0.2 * abs(val)), 4))
        trial_fit = fitness_function(trial, X, y)
        if trial_fit < best_fit:
            best, best_fit = trial, trial_fit
    return best


def fitness_function(tree, X, y):
    """Plain training MSE (lower is better).

    Kept deliberately raw: in this variant the loss is not where the idea
    lives. Skeletons are made comparable by refining their coefficients
    (``tune_constants``) BEFORE this number is trusted, rather than by
    reshaping the loss itself.
    """
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Rank-weighted sampling (``tournament_size`` kept for the contract).

    Parents are drawn with probability proportional to fitness rank —
    softer than a large tournament, so structurally novel skeletons whose
    constants are not yet tuned survive long enough to be refined.
    """
    order = sorted(range(len(population)), key=lambda i: fitnesses[i])
    weights = list(range(len(order), 0, -1))
    picks = random.choices(order, weights=weights, k=n_select)
    return [population[i].copy() for i in picks]


def crossover(parent1, parent2, n_features, max_depth=17):
    """Terminal exchange: trade numeric material, preserve structure.

    A random terminal of a ``parent1`` copy is replaced by a random
    terminal drawn from ``parent2`` — constants and variables migrate
    between lineages while both skeletons survive intact. Structural
    recombination is intentionally absent from the scaffold; structure
    moves through mutation's regrow branch instead.
    """
    child = parent1.copy()
    slots = [(nd, par, i) for nd, par, i in child.get_all_nodes() if nd.is_terminal]
    donors = [nd for nd, _, _ in parent2.get_all_nodes() if nd.is_terminal]
    if not slots or not donors:
        return child
    _, par, idx = slots[random.randint(0, len(slots) - 1)]
    pick = donors[random.randint(0, len(donors) - 1)].copy()
    if par is None:
        return pick
    par.children[idx] = pick
    return child


def mutation(parent, n_features, max_depth=17):
    """Coefficient-first mutation with an occasional structural proposal.

    With probability 0.7 (when constants exist) a single numeric leaf is
    nudged — continuous search. Otherwise a random site is regrown with a
    small fresh subtree — the discrete proposal channel. The two kinds of
    moves are never mixed in one call, keeping the division of labor
    legible in the lineage.
    """
    child = parent.copy()
    leaves = _numeric_leaves(child)
    if leaves and random.random() < 0.7:
        leaf = random.choice(leaves)
        val = float(leaf.value)
        leaf.value = str(round(val + random.gauss(0.0, 0.5 + 0.25 * abs(val)), 4))
        return child
    spots = child.get_all_nodes()
    _, par, idx = spots[random.randint(0, len(spots) - 1)]
    fresh = generate_tree('grow', 3, n_features)
    if par is None:
        return fresh
    par.children[idx] = fresh
    return child


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation with elite coefficient refinement.

    Args:
        population: list of Node trees
        fitnesses: list of float fitness values (lower is better)
        X_train, y_train: training samples
        n_features: number of input features
        pop_size: desired population size
        crossover_rate, mutation_rate: operator probabilities
        max_depth: maximum allowed tree depth

    Returns:
        list of Node — next generation population (length pop_size)

    Only the elite is refined in the scaffold (3 extra evaluations per
    generation). Widening refinement to the top-k, or refining offspring
    before insertion, is the budget-allocation experiment this variant
    exists to run.
    """
    elite_idx = int(np.argmin(fitnesses))
    nxt = [tune_constants(population[elite_idx], X_train, y_train)]

    while len(nxt) < pop_size:
        pair = selection(population, fitnesses, 2)
        u = random.random()
        if u < crossover_rate:
            nxt.append(crossover(pair[0], pair[1], n_features, max_depth))
        elif u < crossover_rate + mutation_rate:
            nxt.append(mutation(pair[0], n_features, max_depth))
        else:
            nxt.append(pair[0])
    return nxt[:pop_size]
