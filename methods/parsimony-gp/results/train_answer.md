When I evolve a population of expression trees for symbolic regression, I keep running into the same failure mode. After a quiet opening in which the mean node count barely moves, the average program size starts climbing generation after generation while fitness stops improving. This is bloat, and I am careful to separate it from legitimate growth: a hard regression target may genuinely need a larger expression. The pathology is growth that buys no fitness. It makes evaluation slower, makes the final expression opaque, and usually makes generalisation worse. The standard remedy is to make large programs pay rent — select on a penalised fitness $f_p(x) = f(x) - c\,\ell(x)$, where $\ell(x)$ is program size and $c$ is a constant pressure against length, with the raw $f$ kept for recognising solutions. This is a sensible soft constraint, in the spirit of an MDL tradeoff, but it leaves the only question that matters unanswered: what should $c$ be? Too small and the population still bloats; too large and GP treats shrinking as its real objective and collapses toward tiny useless programs. The good value depends on the problem, the primitive set, the population, the selection scheme, and it drifts as the run proceeds. Zhang and Mühlenbein's adaptive-coefficient idea shows that moving $c$ over time helps, but it still gives no rule for what value produces a specified effect on mean size — it remains a tuned schedule, not a computed one. Anti-bloat operators, hard size and depth limits, the Tarpeian method, and multi-objective formulations each fight bloat from a different angle, but none of them hands me a single computable dial linked in closed form to the mean-size trajectory. I want to compute the pressure from the current population rather than tune it.

I propose covariant parsimony pressure: a parsimony coefficient that is computed each generation from population statistics already in hand, derived so that it makes selection neutral with respect to the size advantage it is currently exploiting. The starting point is the size evolution equation. For selection followed by symmetric subtree crossover — symmetric meaning crossover does not favour one parent order over the other — the expected next mean size depends only on which length classes selection chooses, $E[\mu(t+1)] = \sum_\ell \ell\, p(\ell,t)$, where $p(\ell,t)$ is the probability that a selection event picks a size-$\ell$ program. If $\Phi(\ell,t)$ is the population fraction at size $\ell$, then $\mu(t) = \sum_\ell \ell\,\Phi(\ell,t)$, so the expected change is $$E[\Delta\mu] = \sum_\ell \ell\,\big(p(\ell,t) - \Phi(\ell,t)\big).$$ This identifies the only lever: crossover reshapes the size distribution but under the symmetry condition does not move the mean in expectation, so the mean moves solely because selection over- or under-samples length classes. To control bloat I must make selection neutral with respect to the length advantage it sees.

To turn that into an equation for fitness I work in fitness-proportionate selection, where $p(\ell,t)$ has a closed form. With $\bar f(\ell,t)$ the average fitness among size-$\ell$ programs and $\bar f(t)$ the population mean fitness, $p(\ell,t) = \Phi(\ell,t)\,\bar f(\ell,t)/\bar f(t)$. Substituting gives $E[\Delta\mu] = \frac{1}{\bar f(t)}\sum_\ell \ell\,\big(\bar f(\ell,t) - \bar f(t)\big)\Phi(\ell,t)$. This is almost a covariance, but covariance wants $\ell - \mu(t)$ rather than $\ell$. I split $\ell = (\ell - \mu(t)) + \mu(t)$; the first part yields $\mathrm{Cov}(\ell,f)$, and the second part carries a factor $\mu(t)\sum_\ell\big(\bar f(\ell,t)-\bar f(t)\big)\Phi(\ell,t) = \mu(t)\,(\bar f(t)-\bar f(t)) = 0$. The cross-term vanishes for a precise reason rather than by hand-waving, leaving $$E[\Delta\mu] = \frac{\mathrm{Cov}(\ell,f)}{\bar f(t)}.$$ This is Price's theorem read in the language of program size: the expected one-generation change in the mean of a heritable feature is its covariance with fitness divided by mean fitness. Operationally, in the maximised-fitness convention a positive covariance between length and fitness means selection has a length advantage to exploit, so the mean grows.

Now I bring back the penalty without assuming it is linear, writing $f_p(x,t) = f(x) - g(\ell(x),t)$ for any generation-dependent size function $g$. The same identity applied to $f_p$ gives $$E[\Delta\mu] = \frac{\mathrm{Cov}(\ell,f-g)}{\bar f - \bar g} = \frac{\mathrm{Cov}(\ell,f) - \mathrm{Cov}(\ell,g)}{\bar f - \bar g}.$$ Setting $E[\Delta\mu] = 0$, and provided the denominator is nonzero, the no-growth condition is simply $$\mathrm{Cov}(\ell,g) = \mathrm{Cov}(\ell,f).$$ The penalty must have exactly the same covariance with size as the raw fitness currently has; at the no-growth point the mean-fitness denominator drops out because I am forcing the numerator to zero. This covariance condition, not the surface form of the penalty, is the real object. For the traditional linear penalty $g(\ell,t) = c(t)\,\ell$, we have $\mathrm{Cov}(\ell,g) = c(t)\,\mathrm{Cov}(\ell,\ell) = c(t)\,\mathrm{Var}(\ell)$, and matching it to $\mathrm{Cov}(\ell,f)$ forces $$c(t) = \frac{\mathrm{Cov}(\ell,f)}{\mathrm{Var}(\ell)}.$$ That is exactly the ordinary-least-squares slope of fitness against size — the per-node fitness advantage selection currently sees — and subtracting that slope times length removes the linear size-fitness advantage from the selection signal. It is dynamic because both the numerator and denominator drift through the run, which is why it must be recomputed every generation; a single constant cannot track it.

The same one-unknown equation handles the obvious variants and confirms I am not merely fitting a special case. A power penalty $g(\ell,t) = c(t)\,\ell^k$ gives $c(t) = \mathrm{Cov}(\ell,f)/\mathrm{Cov}(\ell,\ell^k)$, reducing to the linear formula at $k=1$; a centred linear penalty $g = c(t)(\ell - \mu(t))$ gives $\mathrm{Cov}(\ell,\mu(t)) = 0$ since $\mu(t)$ is constant across the population, recovering $c(t)\,\mathrm{Var}(\ell)$ again. Freezing size is only one target. To make the expected mean follow a chosen trajectory $\gamma$, I impose $E[\mu(t+1)] = \gamma(t+1)$; with $\delta = \gamma(t+1) - \mu(t)$ and $g = c(t)\ell^k$ this solves to $c(t) = \big(\mathrm{Cov}(\ell,f) - \delta\,\bar f\big)/\big(\mathrm{Cov}(\ell,\ell^k) - \delta\,E[\ell^k]\big)$, which for $k=1$ is $\big(\mathrm{Cov}(\ell,f) - \delta\,\bar f\big)/\big(\mathrm{Var}(\ell) - \delta\,\mu(t)\big)$, collapsing to $\mathrm{Cov}(\ell,f)/\mathrm{Var}(\ell)$ when $\delta = 0$. Anchoring the setpoint to the initial mean, $\gamma(t+1) = \mu(0)$, gives $c(t) = \big(\mathrm{Cov}(\ell,f) - (\mu(0)-\mu(t))\bar f\big)/\big(\mathrm{Cov}(\ell,\ell^k) - (\mu(0)-\mu(t))E[\ell^k]\big)$, which adds a restoring term: because the zero-growth equation controls the mean only in expectation, finite-population sampling noise can let the realised mean drift, and anchoring to $\mu(0)$ pushes it back. I keep the selection-scheme boundary explicit. The exact covariance derivation uses fitness-proportionate selection; the size evolution equation itself is broader since it is written in terms of $p(\ell,t)$. Tournament selection does not give the same simple $p(\ell,t)$, so the coefficient is not an exact theorem for every tournament — but the ratio still estimates the linear size-fitness slope in the current population, and tournament selection amplifies the same ordering, so cancelling that slope is the natural practical coefficient and it carries over without changing the operator.

Translating to the implementation means respecting a sign flip. The derivation uses maximised fitness and writes $f - c\ell$; the scaffold here uses mean squared error, where lower is better, so making large trees worse means selecting on $\text{penalized} = \text{MSE} + c\,\ell$. This matches gplearn's convention, where `_Program.fitness()` returns `raw_fitness_ - c*length*metric.sign` and $\text{metric.sign} = -1$ for MSE, yielding raw error plus $c$ times length. I preserve gplearn's automatic rule verbatim at the measurement step, `np.cov(length, fitness)[1, 0] / np.var(length)`, guard the degenerate zero-variance case, and clamp the coefficient to $[0, 0.001]$: the lower clamp prevents a negative $c$ from rewarding length under the lower-better convention, and the upper clamp keeps one noisy generation from making selection ignore fit. The raw MSE remains the truth for elitism, stopping, and reporting; the penalty is a parent-selection signal only. The per-generation loop is then simple — keep the raw-MSE elite, form the penalised vector $\text{MSE} + c\,\ell$, and use that vector solely inside tournament selection while crossover, mutation, and reproduction fill the rest of the population. The size-control logic is just the few lines that compute $c$ and build the penalised vector; everything else is the generic GP machinery.

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
