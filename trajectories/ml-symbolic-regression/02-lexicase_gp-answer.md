**Problem.** Standard GP selects on the *aggregate* MSE — it crushes each tree's per-case error vector
into one scalar, deleting *which* regions a tree is good at and burying specialists that carry the
building blocks. On the transcendental benchmarks this produced premature convergence onto a wrong form:
Nguyen-7 had a −1.0 blow-up seed (mean 0.330) and Nguyen-10 declined across seeds (mean 0.588). The
lever is selection — change what it rewards.

**Key idea (ε-lexicase selection).** Do not aggregate — *filter*. For each parent, shuffle the training
cases into a random order and walk them as a chain of elitism gates: at each case keep only the
individuals whose error is within `ε` of the best, until one remains (or cases run out → random
survivor). Difficult cases land early often and slice the pool to the specialist on that hard case;
each parent comes down a different conjunction of cases, so the population spreads across behavior space
instead of collapsing onto one lineage.

**Why these choices.** Exact-best gates are measure-zero on continuous errors (the first gate would
empty the pool to one), so the gate is relaxed to `e_t(i) ≤ e*_t + ε`. A *fixed* `ε` is a brittle,
problem-dependent knob that goes blind late in a run, so `ε_t` is set automatically from the robust
spread of the case's errors: `ε_t = MAD(e_t)`. MAD, not standard deviation — a GP population is full of
blow-up outliers that would inflate σ and pass everyone; MAD's 50% breakdown point ignores them. Both
the pool elite `e*_t` and `ε_t` are recomputed over the **current candidate pool** at each gate
(pool-relative / dynamic form), so the band keeps discriminating as the pool shrinks and never empties.
Crossover and mutation are unchanged standard subtree operators — selection only is changed.
`fitness_function` stays raw MSE (the loop tracks the best-ever tree and elitism by it). If the per-case
error matrix is absent, `selection` falls back to a size-7 tournament.

**Hyperparameters.** `ε` = pool MAD per case (no tuned constant); tournament fallback size 7;
crossover_rate = 0.9; mutation_rate = 0.05; elitism = 1; depth cap = 8; pop_size = 500; generations =
50. Selection cost is `O(P²N)` worst case (vs tournament's `O(Pk)`).

**What to watch.** If selection diversity is the bottleneck, Nguyen-7's −1.0 seed should disappear (all
seeds positive, mean climbing toward ~0.97) and Nguyen-10 should rise above 0.588 by preserving the
sin/cos specialists tournament buried; Koza-3 (already solved) should stay near 0.99. A large wall-clock
jump is expected from the `O(P²N)` selection — the question is whether it buys back the lost seeds.

```python
def fitness_function(tree, X, y):
    """MSE fitness — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def _per_case_errors(population, X, y):
    """Compute per-case absolute errors for the entire population.

    Returns:
        numpy array of shape (len(population), n_samples)
    """
    errors = np.empty((len(population), X.shape[0]))
    for i, tree in enumerate(population):
        y_pred = safe_evaluate(tree, X)
        errors[i] = np.abs(y - y_pred)
    return errors


def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
    """Epsilon-lexicase selection.

    Requires _errors (per-case errors), _X, _y to be passed via
    evolve_one_generation. Falls back to tournament if not available.
    """
    selected = []
    pop_size = len(population)

    if _errors is None:
        # Fallback to tournament
        for _ in range(n_select):
            candidates = random.sample(range(pop_size), min(7, pop_size))
            best = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best].copy())
        return selected

    n_cases = _errors.shape[1]
    for _ in range(n_select):
        candidates = list(range(pop_size))
        cases = list(range(n_cases))
        random.shuffle(cases)

        for case in cases:
            if len(candidates) <= 1:
                break
            case_errors = _errors[candidates, case]
            # Semi-dynamic epsilon-lexicase (La Cava 2016/2019): candidates
            # survive iff their error ≤ best_on_case + MAD. The previous
            # `median + MAD` admitted most of the population and degraded
            # lexicase toward random selection.
            min_err = float(np.min(case_errors))
            mad = float(np.median(np.abs(case_errors - float(np.median(case_errors)))))
            candidates = [c for c, e in zip(candidates, case_errors) if e <= min_err + mad]

        winner = random.choice(candidates)
        selected.append(population[winner].copy())

    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Standard subtree crossover."""
    offspring = parent1.copy()
    donor = parent2.copy()

    off_size = offspring.size()
    don_size = donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)

    donor_nodes = donor.get_all_nodes()
    donor_subtree = donor_nodes[don_point][0].copy()

    off_nodes = offspring.get_all_nodes()
    node, parent, child_idx = off_nodes[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    if offspring.depth() > max_depth:
        return parent1.copy()

    return offspring


def mutation(parent, n_features, max_depth=17):
    """Subtree mutation — replace a random subtree with a new random tree."""
    offspring = parent.copy()
    tree_size = offspring.size()
    if tree_size <= 1:
        return generate_tree('grow', 3, n_features)

    mut_point = random.randint(1, tree_size - 1)
    new_subtree = generate_tree('grow', 3, n_features)

    nodes = offspring.get_all_nodes()
    node, par, child_idx = nodes[mut_point]
    if par is not None:
        par.children[child_idx] = new_subtree
    else:
        offspring = new_subtree

    if offspring.depth() > max_depth:
        return parent.copy()

    return offspring


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Lexicase GP generation — uses epsilon-lexicase selection."""
    new_population = []

    # Elitism: keep best
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    # Pre-compute per-case errors for lexicase selection
    errors = _per_case_errors(population, X_train, y_train)

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2, _errors=errors)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1, _errors=errors)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, fitnesses, 1, _errors=errors)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```
