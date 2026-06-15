**Problem.** Lexicase cured premature convergence (Nguyen-7's −1.0 seed gone, mean 0.330 → 0.968) but
left tree *size* completely unmanaged — by rewarding case specialists it keeps large, contorted trees,
and over 50 generations the population bloats with inert subexpressions. Bloat slows evaluation, hurts
readability, and overfits the training sample, which capped Koza-3 (0.922) and Nguyen-10 (0.714). The
lever is complexity control.

**Key idea (parsimony-pressured GP).** Select on a penalized fitness `penalized = MSE + c·l` (size `l`),
with the coefficient *computed*, not tuned: `c = Cov(length, MSE)/Var(length)`. This is Price's-theorem
exact — the expected one-generation change in mean size is `Cov(l, f)/f̄`, so subtracting the OLS slope
of fitness on size from selection cancels the size advantage selection is currently exploiting and sets
expected size growth to zero. The penalty steers selection only; raw MSE remains the truth for elitism
and reporting.

**Why these choices.** The coefficient `c = Cov(l,f)/Var(l)` falls out of the size-evolution equation:
under symmetric subtree crossover the mean moves only through length-class selection, so cancelling the
size-fitness slope freezes expected size — no per-benchmark constant to babysit. The derivation is exact
for fitness-proportionate selection; the size-7 tournament kept here does not give the exact theorem, but
the same ratio still estimates the slope and tournament amplifies the same ordering, so it carries over
without changing the operator. Conventions: the scaffold is lower-is-better, so the penalty is **added**
(`MSE + c·l`) and clamped to `[0, 0.001]` — the lower clamp forbids a negative `c` that would reward
bloat, the upper clamp stops a noisy generation from making selection ignore fit and collapse to stubs.
The penalty is applied in **selection only**; `fitness_function` returns raw MSE and elitism preserves
the raw-MSE best, so the loop never reports a tree chosen for being small. Crossover and mutation are
unchanged standard subtree operators.

**Hyperparameters.** `c` = clamp(Cov(len, MSE)/Var(len), 0, 0.001), recomputed per generation;
tournament_size = 7; crossover_rate = 0.9; mutation_rate = 0.05; elitism = 1 (raw MSE); depth cap = 8;
pop_size = 500; generations = 50. Selection is `O(Pk)` — far cheaper than lexicase's `O(P²N)`.

**What to watch.** If bloat was the cap, Koza-3 should recover toward ~0.99 (introns removed), Nguyen-10
should rise above 0.714 with a higher floor (tighter, more generalizable sin·cos), and Nguyen-7 should
hold near/above 0.985 (compact trees extrapolate more gracefully than bloated ones) — all at a fraction
of lexicase's wall-clock. Nguyen-10 staying near 0.7 while Koza-3 recovers would mean its bottleneck is
representational reach, not bloat.

```python
def fitness_function(tree, X, y):
    """Raw MSE fitness — lower is better.

    Parsimony pressure is applied at the population level inside
    evolve_one_generation, not here. This ensures best_tree_ever
    in the main loop tracks the best-fitting tree by actual MSE.
    """
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection on (possibly penalized) fitnesses."""
    selected = []
    pop_size = len(population)
    for _ in range(n_select):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())
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
    """Parsimony GP generation with parsimony pressure for bloat control.

    Uses gplearn-style auto parsimony coefficient computed per generation:
        c = Cov(length, fitness) / Var(length)
    clamped to [0, 0.001] to prevent runaway penalization.
    Parsimony pressure is applied only during selection; elitism uses
    raw fitness so the best-fitting individual is always preserved.
    """
    new_population = []

    # Adaptive parsimony coefficient (gplearn 'auto' method, clamped)
    lengths = np.array([tree.size() for tree in population], dtype=float)
    raw_fit = np.array(fitnesses, dtype=float)
    len_var = float(np.var(lengths))
    if len_var > 1e-15:
        parsimony_coeff = float(np.cov(lengths, raw_fit)[1, 0]) / len_var
        parsimony_coeff = max(parsimony_coeff, 0.0)
        parsimony_coeff = min(parsimony_coeff, 0.001)
    else:
        parsimony_coeff = 0.0

    # Penalized fitnesses for selection only
    penalized = [f + parsimony_coeff * l for f, l in zip(fitnesses, lengths)]

    # Elitism: keep best by raw fitness (not penalized)
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, penalized, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, penalized, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, penalized, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```
