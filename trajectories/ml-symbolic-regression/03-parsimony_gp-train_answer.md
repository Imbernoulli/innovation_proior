Lexicase half-confirmed the bet and half-refuted it. What I most wanted is there: Nguyen-7's $-1.0$ catastrophe is gone, all three seeds positive (0.986, 0.946, 0.972), the mean up from 0.330 to 0.968 — the premature-convergence cure working as predicted. But Koza-3, the control that was already solved, actually *slipped* on its two harder seeds (0.908, 0.859; mean 0.922, down from 0.993), and Nguyen-10 only rose to 0.714 with the floor still at a 0.500 seed. I read the Koza-3 slip and the Nguyen-10 shortfall together, because they share one cause. Lexicase fixed *diversity* — it stops the population collapsing — but does nothing about *size*, and by rewarding specialists on individual cases it actively keeps large, contorted trees that happen to nail one nasty case. Over 50 generations the mean tree size drifts up: programs accumulate inert subexpressions, "introns," because near a fitness plateau larger trees are more likely to survive crossover unharmed (a random crossover point lands in the padding, not the working core), so selection quietly rewards size for its protective value. Bloat slows evaluation, makes the final expression unreadable, and — the part the numbers show — *hurts generalization*: a bloated tree threads the training points with high-order wiggle and misses between them, exactly the Koza-3 slip (20 train, 100 test) and the Nguyen-10 shortfall (100 train, 400 test). The lever is no longer selection-for-diversity; it is complexity control.

I propose parsimony-pressured GP, where selection acts on a penalized fitness $f_p(x) = \text{MSE}(x) + c\cdot l(x)$ with $l(x)$ the tree size, and the crucial move is that the coefficient $c$ is *computed* from the current population, not tuned. "Penalize big trees" is easy to say and easy to get wrong: too small a $c$ and bloat continues, too large and GP treats shrinking as the real objective and collapses toward useless stubs, and the right value changes with the problem, the primitives, the population, the selection scheme, and even the generation. So I derive the coefficient from what actually determines the expected change in mean size. The size-evolution equation is the starting point: for selection followed by symmetric subtree crossover — crossover that does not favor either parent order — the expected next mean size depends only on which length classes selection picks,
$$E[\mu(t{+}1)] = \sum_l l\,p(l,t), \qquad E[\Delta\mu] = \sum_l l\,\big(p(l,t) - \Phi(l,t)\big),$$
where $\Phi(l,t)$ is the current population fraction at size $l$ and $p(l,t)$ the probability a selection event chooses a size-$l$ program. This already names the lever: under the symmetry condition crossover does not move the mean in expectation — the mean moves *only* because selection over- or under-samples length classes. To stop bloat I must make selection neutral with respect to the length advantage it is currently exploiting.

To turn that into an equation for fitness I first use fitness-proportionate selection, where $p(l,t)$ has a closed form, $p(l,t) = \Phi(l,t)\,\bar f(l,t)/\bar f(t)$, with $\bar f(l,t)$ the mean fitness among size-$l$ programs. Substituting gives $E[\Delta\mu] = \frac{1}{\bar f}\sum_l l\,(\bar f(l,t)-\bar f(t))\,\Phi(l,t)$. This is almost a covariance, except a covariance wants $l-\mu$, not $l$. Splitting $l=(l-\mu)+\mu$: the first piece gives $\text{Cov}(l,f)$; the second is $\mu\sum_l(\bar f(l,t)-\bar f(t))\Phi(l,t)$, and that population-weighted deviation of size-class fitness from the population mean is $\bar f - \bar f = 0$. The cross-term vanishes for a precise reason, and I am left with Price's theorem in the language of program size,
$$E[\Delta\mu] = \frac{\text{Cov}(l,f)}{\bar f}.$$
The operational reading is the whole point: a positive covariance between length and fitness means selection has a length advantage to exploit, so the mean grows — that is bloat, named exactly.

Now bring back the penalty without assuming it is linear, $f_p(x,t) = f(x) - g(l(x),t)$ for any size function $g$. The same identity gives $E[\Delta\mu] = \text{Cov}(l, f-g)/(\bar f - \bar g) = (\text{Cov}(l,f) - \text{Cov}(l,g))/(\bar f - \bar g)$. For *no* expected growth I set $E[\Delta\mu]=0$, and provided the denominator is nonzero the condition is simply $\text{Cov}(l,g) = \text{Cov}(l,f)$ — the mean-fitness denominator drops out because I force the numerator to zero. Taking the traditional linear penalty $g = c\cdot l$ gives $\text{Cov}(l,g) = c\,\text{Var}(l)$, and setting that equal to $\text{Cov}(l,f)$ forces
$$c = \frac{\text{Cov}(l, f)}{\text{Var}(l)}.$$
That is the coefficient for zero expected size change. It is dynamic, since both pieces change during the run, and it reads cleanly: it is the ordinary-least-squares slope of fitness against size, the per-node fitness advantage selection currently sees, and subtracting that slope times length removes the linear size-fitness advantage from the selection signal. No tuning, no per-benchmark constant — the population measures the pressure it needs.

I keep the selection-scheme boundary honest. The exact covariance derivation assumes fitness-proportionate $p(l,t)$; this rung keeps the size-7 tournament, which does not give that closed form, so $c = \text{Cov}(l,f)/\text{Var}(l)$ is not an exact theorem here. But the ratio still *estimates* the linear size-fitness slope in the current population, and tournament selection amplifies the same ordering — it just sharpens the preference for whoever has the better penalized value. Cancelling that slope is therefore the natural practical coefficient, and it carries over without changing the operator, which matters because I keep the standard uniform-point subtree crossover and mutation from the previous rungs. I change *only* the number selection sees.

Translating to the scaffold's conventions is where the signs and clamps live. The theory used maximized fitness and wrote $f - c\cdot l$; the scaffold gives MSE, where *lower* is better, so making large trees worse means selecting on $\text{penalized} = \text{MSE} + c\cdot l$ — the penalty is *added*, and selection minimizes. I compute $c = \text{Cov}(\text{length}, \text{raw MSE})/\text{Var}(\text{length})$ from the current lengths and the raw-MSE vector, guarding the zero-variance case with $c=0$, then clamp to $[0, 0.001]$. The lower clamp matters under lower-is-better: a *negative* coefficient would reward length, since subtracting a negative adds to bad trees' favor, so I floor at 0 and never reward bloat. The upper clamp at 0.001 keeps one noisy generation, where a spurious size-fitness correlation spikes $c$, from making selection ignore fit entirely and collapsing the population to stubs. Critically, the penalty lives in *selection only*: `fitness_function` returns raw MSE unchanged, the elite carried forward each generation is the raw-MSE best, and only the penalized vector $\text{MSE}+c\cdot l$ is handed to the tournament. Baking the penalty into the fitness would make the loop preserve and report a tree chosen for being small rather than for fitting — exactly the collapse the upper clamp guards against.

The falsifiable expectations against the lexicase numbers. The mechanism is the OLS slope of fitness on size cancelled out of selection, so the direct prediction is that mean tree size stops drifting up, and the downstream prediction the leaderboard sees is that controlling bloat recovers the generalization bloat was costing. So Koza-3 should come *back up* from 0.922 toward 0.99 and beyond — the slip was overfitting introns, and removing them should restore the clean polynomial fit; if Koza-3 does not recover, the bloat story is wrong. Nguyen-10 should improve on 0.714, because the 0.643 and 0.500 seeds were, on this reading, bloated overfits of the $\sin\!\cdot\!\cos$ product, and a parsimony-pressured search should find a tighter, more generalizable form and lift the floor. Nguyen-7 is the interesting one: lexicase already cured its catastrophe and parsimony does not carry lexicase's diversity machinery, so the open question is whether plain tournament plus parsimony can hold it near 0.97. My expectation is yes — the $-1.0$ blow-up was a divergence problem as much as a diversity one, and a compact tree is *less* prone to diverging off the test range than a bloated one, so Nguyen-7 should stay near or above 0.985 with smaller trees extrapolating more gracefully. If all three hold, this rung sits clearly above lexicase on the overall mean and does so *fast*, since parsimony keeps trees small and $O(Pk)$ tournament selection is orders of magnitude cheaper than lexicase's $O(P^2N)$. The one outcome that would refute the chain is Nguyen-10 staying stuck near 0.7 while Koza-3 recovers — that would mean bloat was never the Nguyen-10 bottleneck and the cap there is raw representational reach, which no selection-side fix can lift.

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
