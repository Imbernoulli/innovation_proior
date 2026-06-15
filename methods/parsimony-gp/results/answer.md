# Covariant parsimony pressure

Compute the size pressure from the current population instead of tuning it as a constant. In the maximised-fitness derivation, use `f_p(x,t) = f(x) - g(l(x),t)`. The size evolution equation plus fitness-proportionate selection gives:

```text
E[Delta mu] = Cov(l, f) / fbar(t)
```

The covariance step is valid because the extra `mu(t)` term is:

```text
mu(t) * sum_l (fbar(l,t) - fbar(t)) Phi(l,t) = 0
```

With a size penalty:

```text
E[Delta mu] = (Cov(l,f) - Cov(l,g)) / (fbar - gbar)
```

So no expected growth requires:

```text
Cov(l,g) = Cov(l,f)
```

For the linear penalty `g = c(t) * l`:

```text
c(t) = Cov(l,f) / Var(l)
```

That is the OLS slope of fitness on size. It must be recomputed each generation because both the numerator and denominator drift.

## Extensions

Power penalty:

```text
g = c(t) * l^k
c(t) = Cov(l,f) / Cov(l,l^k)
```

Target tracking, with `E[mu(t+1)] = gamma(t+1)` and `delta = gamma(t+1) - mu(t)`:

```text
c(t) = (Cov(l,f) - delta*fbar) / (Cov(l,l^k) - delta*E[l^k])
```

For `k = 1`:

```text
c(t) = (Cov(l,f) - delta*fbar) / (Var(l) - delta*mu(t))
```

Anchoring the target to the initial mean, `gamma(t+1) = mu(0)`, gives:

```text
c(t) = (Cov(l,f) - (mu(0)-mu(t))*fbar)
       / (Cov(l,l^k) - (mu(0)-mu(t))*E[l^k])
```

This counters finite-population drift around the zero-growth expectation.

## Implementation notes

gplearn's automatic coefficient is exactly:

```python
np.cov(length, fitness)[1, 0] / np.var(length)
```

gplearn then applies the sign convention inside `_Program.fitness()` as `raw_fitness_ - c * length * metric.sign`. For lower-is-better MSE, `metric.sign = -1`, so the parent-selection score is:

```text
penalized = MSE + c * length
```

The penalty is for selection only. Elitism, stopping, and best-program reporting stay on raw MSE.

## MLS-Bench code

```python
import random
import numpy as np


def fitness_function(tree, X, y):
    """Raw objective: MSE, lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection on the provided lower-better fitness vector."""
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
    off_size, don_size = offspring.size(), donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)
    donor_subtree = donor.get_all_nodes()[don_point][0].copy()
    node, parent, child_idx = offspring.get_all_nodes()[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    if offspring.depth() > max_depth:
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    """Standard subtree mutation."""
    offspring = parent.copy()
    tree_size = offspring.size()
    if tree_size <= 1:
        return generate_tree('grow', 3, n_features)

    mut_point = random.randint(1, tree_size - 1)
    new_subtree = generate_tree('grow', 3, n_features)
    node, par, child_idx = offspring.get_all_nodes()[mut_point]
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
    """Produce one generation with covariance-based size pressure."""
    lengths = np.array([tree.size() for tree in population], dtype=float)
    raw_fit = np.array(fitnesses, dtype=float)
    length_var = float(np.var(lengths))

    if length_var > 1e-15:
        auto_c = float(np.cov(lengths, raw_fit)[1, 0]) / length_var
        c = min(max(auto_c, 0.0), 0.001)
    else:
        c = 0.0

    penalized = [f + c * l for f, l in zip(fitnesses, lengths)]

    new_population = []
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, penalized, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parent = selection(population, penalized, 1)[0]
            child = mutation(parent, n_features, max_depth)
        else:
            child = selection(population, penalized, 1)[0]
        new_population.append(child)

    return new_population[:pop_size]
```
