# Anytime-schedule state, advanced once per evolve_one_generation call.
# ``immigrant_frac`` is the initial share of the population replaced by
# fresh random trees; it decays linearly to zero by ``horizon`` so that
# exploration is front-loaded and the late budget is spent exploiting.
_SCHEDULE = {"generation": 0, "immigrant_frac": 0.2, "horizon": 25}


def fitness_function(tree, X, y):
    """Plain training MSE (lower is better).

    Every call to this function is one tick of the evaluation meter this
    variant treats as the scarce resource. The scaffold keeps the loss
    itself trivial; the variant's content is WHERE the calls are spent
    (see the schedule in ``evolve_one_generation``), not how each call is
    scored. Caching by expression string is a natural way to stop paying
    twice for duplicate genotypes.
    """
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Truncation selection: parents come only from the top quartile.

    Cheap and strongly exploitative — the exploration this removes is
    reinjected explicitly (and on a decaying schedule) by the immigrant
    stream, which makes the exploration budget visible and tunable instead
    of being hidden inside selection pressure. ``tournament_size`` is kept
    for the interface.
    """
    order = sorted(range(len(population)), key=lambda i: fitnesses[i])
    cut = max(1, len(order) // 4)
    pool = order[:cut]
    return [population[random.choice(pool)].copy() for _ in range(n_select)]


def crossover(parent1, parent2, n_features, max_depth=17):
    """Headless-chicken graft: splice a fresh random subtree into parent1.

    ``parent2`` is intentionally ignored by the scaffold: a graft from a
    random tree is the cheapest structural macro-move available, and under
    an anytime schedule its usefulness should fade as the meter empties —
    replacing it with true recombination late in the run is one of the
    scheduling experiments this variant invites.
    """
    host = parent1.copy()
    spots = host.get_all_nodes()
    _, par, idx = spots[random.randint(0, len(spots) - 1)]
    fresh = generate_tree('grow', 3, n_features)
    if par is None:
        host = fresh
    else:
        par.children[idx] = fresh
    if host.depth() > max_depth:
        return parent1.copy()
    return host


def mutation(parent, n_features, max_depth=17):
    """Subtree regeneration at a random site (small 'grow' replacement)."""
    mutant = parent.copy()
    spots = mutant.get_all_nodes()
    _, par, idx = spots[random.randint(0, len(spots) - 1)]
    regrown = generate_tree('grow', 2, n_features)
    if par is None:
        return regrown
    par.children[idx] = regrown
    return mutant


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Create the next generation on a decaying exploration schedule.

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

    Slot budget per generation: 1 elite, a decaying block of random
    immigrants (exploration, front-loaded), and the remainder bred from
    the truncated parent pool (exploitation, grows over time). By the
    schedule horizon the immigrant stream is dry and the whole budget
    exploits. The elite guarantees the anytime property: the best-so-far
    answer is never lost between meter readings.
    """
    gen = _SCHEDULE["generation"]
    _SCHEDULE["generation"] = gen + 1
    decay = max(0.0, 1.0 - gen / float(_SCHEDULE["horizon"]))
    n_immigrants = int(pop_size * _SCHEDULE["immigrant_frac"] * decay)

    nxt = [population[int(np.argmin(fitnesses))].copy()]
    for _ in range(n_immigrants):
        nxt.append(generate_tree('grow', 4, n_features))

    while len(nxt) < pop_size:
        mom, dad = selection(population, fitnesses, 2)
        toss = random.random()
        if toss < crossover_rate:
            nxt.append(crossover(mom, dad, n_features, max_depth))
        elif toss < crossover_rate + mutation_rate:
            nxt.append(mutation(mom, n_features, max_depth))
        else:
            nxt.append(mom)
    return nxt[:pop_size]
