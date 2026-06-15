**Problem.** Symbolic regression: recover a readable formula fitting `(X, y)` — both the functional
*form* and its constants. The candidates are expression trees of unrestricted size and shape, so there
is no parameter vector and no gradient, only the ability to *score* a candidate. The scaffold's default
fill does no search (random selection, copy-only operators); the floor is to turn the contract into a
genuine Darwinian loop over trees.

**Key idea (standard, Koza-style GP).** Let the individual *be* the expression tree and recombine by
**subtree exchange** — since any subtree is a valid expression, every swap yields a valid offspring.
Each generation: score every tree by raw MSE, carry the best forward (elitism), then breed by
tournament selection + subtree crossover (0.9) + subtree mutation (0.05) + reproduction (remainder).
The best tree ever seen is the discovered model.

**Why these choices.** Fitness is raw **MSE** (lower better) and stays free of any penalty, because the
loop tracks the best tree *by this number*. Selection is a **size-7 tournament**: it compares ranks
only, so it is invariant to fitness scale/spread — no single super-fit tree swamps the pool early, and
pressure is set by `k` not by score compression late, the two failure modes of fitness-proportionate
selection. Crossover uses **uniform** crossover points (the plainest Koza form — no function/terminal
weighting at this rung) and a **depth cap** rejecting over-deep offspring (the bloat brake). Mutation
grafts a freshly grown subtree to reintroduce material crossover alone cannot. Elitism guarantees the
best-of-generation never worsens.

**Hyperparameters.** tournament_size = 7; crossover_rate = 0.9; mutation_rate = 0.05; reproduction =
remainder (0.05); elitism = 1; offspring depth cap = 8 (init depth 6 + 2); fresh-subtree depth = 3;
pop_size = 500; generations = 50.

**What to watch.** Koza-3 (a reachable polynomial) should be steady and high; the transcendental
benchmarks (Nguyen-7 logs + extrapolation, Nguyen-10 sin·cos) should show **high seed-to-seed variance**
from premature convergence — a good mean dragged down by seeds that lock onto the wrong form early, and
on the extrapolating Nguyen-7 a wrong form can drive R² sharply negative (floored at 0). That failure is
what forces selection *shaping* at the next rung.

```python
def fitness_function(tree, X, y):
    """MSE fitness — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection."""
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

    # Pick random crossover points
    off_size = offspring.size()
    don_size = donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)

    # Extract donor subtree
    donor_nodes = donor.get_all_nodes()
    donor_subtree = donor_nodes[don_point][0].copy()

    # Replace in offspring
    off_nodes = offspring.get_all_nodes()
    node, parent, child_idx = off_nodes[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    # Reject if too deep
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
    """Standard GP generation with elitism."""
    new_population = []

    # Elitism: keep best
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, fitnesses, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```
