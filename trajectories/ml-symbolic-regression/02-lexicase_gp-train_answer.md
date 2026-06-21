Standard GP did exactly what I expected, and the numbers tell one clean story. Koza-3, the reachable polynomial, is solved across all three seeds (0.985, 0.998, 0.994; mean 0.993). The trouble is entirely on the two transcendental benchmarks, and it is *variance*, not a uniformly low mean. Nguyen-7 has two seeds near-solved (0.998, 0.991) and then seed 456 at $-1.0$ — a complete blow-up where a wrong form found early diverged off the wider test range, drove R² negative, and hit the floor — dragging the mean to 0.330. Nguyen-10 is the same disease in milder, more pervasive form: 0.884, 0.557, 0.322, mean 0.588, a steady decline as the search keeps locking onto partial fits of the $\sin\!\cdot\!\cos$ product. This is the premature-convergence signature, and the lever the diagnosis points at is selection: the failure is *which* lineage wins.

The flaw is structural, not a matter of tuning $k$. Each individual carries a whole vector of per-case errors, one $e_t(i) = |y_t - \hat y_t(i)|$ for each of the $N$ training points, and `fitness_function` crushes that vector into a single scalar mean before selection ever sees it. But the entire premise of GP is that a full solution is *assembled* from partial ones — subprograms that each get some region right, recombined until one tree gets all of it right. Averaging the error vector deletes *which* regions a tree is good at. Two trees with identical mean error can have opposite profiles: one mediocre everywhere, the other the population's *best* on a cluster of hard cases and bad elsewhere. The second is a building block worth gold; tournament cannot tell them apart, and worse, it actively buries the specialist, whose few bad cases inflate its mean so it loses to any generalist. On Nguyen-10 this is exactly the bind — the search needs a tree that nails $\sin(x)$ and another that nails $\cos(y)$, but each specialist reads as mediocre on the mean and is selected out before crossover can splice them. The information I most need to escape premature convergence is the *shape* of the error vector, and aggregation throws it away first.

So I propose $\varepsilon$-lexicase selection, whose governing principle is: do not aggregate — *filter*. Treat each training case as a gate. Start with the whole population as the candidate pool, look at one case, and discard everyone who is not near-best on it; among the survivors look at another case and discard everyone not near-best on *that*; repeat. Each gate is the simplest possible test — "are you still elite, on this case?" — with no averaging anywhere, just a chain of elitism filters, and the individual that survives a long chain is elite on a *conjunction* of cases, exactly the combination no sum could express. The order matters enormously, so I shuffle the cases freshly for *every* parent-selection event; then the first gate, which acts on the whole population and has the most filtering power, is a different case each time. The beautiful consequence is that a case's filtering strength is proportional to its *difficulty*: an easy case almost everyone ties on barely shrinks the pool, while a hard case where only a few reach the best error slices the pool down to those few and hands selection straight to the specialist on that hard case. Difficult cases automatically exert more pressure toward exactly the specialists I want to propagate, and because each parent descends a different random conjunction, the population spreads across behavior space instead of collapsing onto one lineage — the direct antidote to the convergence that produced the $-1.0$ seed.

The structure is right, but dropped onto continuous, real-valued, noisy regression it breaks at the first gate. "Best error on this case" is achieved by exactly one individual, because the chance two distinct trees produce the *identical* floating-point error is essentially zero. So the very first gate slams the pool from 500 to 1, every selection is decided by a single noisy case, and the whole conjunction mechanism is gone — in fact worse than tournament. The mechanism is sound; the *pass condition* is too sharp for a continuous space. In a discrete space many individuals share the best, so the gate keeps a meaningful subset; "exactly equal" is measure-zero here. I do not want exactly-best, I want *near*-best — close enough that I do not disqualify a perfectly good near-elite for losing by $10^{-7}$. So I relax the gate: an individual passes case $t$ if
$$e_t(i) \le e^*_t + \varepsilon,$$
where $e^*_t$ is the best error on that case. Now near-elites survive, a single case no longer empties the pool, and the conjunction behavior returns.

What is $\varepsilon$? A user-set constant is a trap. A fixed band is blind to how the population is doing — early in a run errors are huge and spread wide, late they are small and tight, so a constant reasonable early becomes, late, so wide relative to the now-tiny spread that everyone passes every case, the gates stop filtering, and selection goes random exactly when I should be fine-tuning. And the right value is problem-dependent; a band sensible on Koza-3's error scale would over- or under-filter on Nguyen-10's. So $\varepsilon$ must set itself from the data, tracking the *spread* of the errors on each case: tightly clustered errors need a small band to separate near-elites, a hard contentious case with widely spread errors admits more contenders with a larger band, and as the population solves a case the band shrinks with the errors — the self-scheduling a constant lacks.

The dispersion measure cannot be the textbook standard deviation, and the reason is specific to GP. A GP population at any moment is full of garbage — freshly mutated junk, expressions that divide by near-zero or blow up — and those extreme outliers dominate $\sigma$: a handful of trees at error $10^6$ while everyone real sits near 0.3 sends $\sigma$ through the roof, so a $\sigma$-band would be set by the junk, be enormous, and pass everyone. I need a measure that ignores the tails and reports the typical spread among real contenders. That is the median absolute deviation,
$$\varepsilon_t = \text{MAD}(e_t) = \operatorname{median}_j\big|\,e_t(j) - \operatorname{median}_k e_t(k)\,\big|.$$
The inner median is the typical error, the outer the typical distance from it; a minority of arbitrarily huge outliers moves it by at most one rank step, because MAD has a 50% breakdown point — it simply does not see the junk. So $\varepsilon_t = \text{MAD}(e_t)$ reports the scale the real contenders live in, exactly where "near-elite" should be judged. It is parameter-free, auto-scales to each case's difficulty, and shrinks as the population converges. That MAD-not-$\sigma$ choice is forced by the outlier-heavy nature of GP populations specifically.

One refinement on the reference set. The cheap version computes $e^*_t$ and $\varepsilon_t$ once per generation over the whole population and freezes them. But by the third or fourth gate the pool is a handful of strong individuals while "best" is still measured against the original 500, most already filtered out. So I recompute *both* the pool elite $e^*_t$ and the MAD over the *current* candidate pool at each gate. As the pool homogenizes the pool-MAD collapses, so $\varepsilon$ shrinks further and the later gates keep filtering hard instead of waving everyone through, and because the pool-best always clears its own band the pool never empties. This pool-relative, dynamic form keeps the gates discriminating all the way down the chain. The cost stays in lexicase's class — recomputing a minimum and a MAD over the current pool at each gate is linear in the pool size per case, so one selection is $O(PN)$ worst case and filling the population is $O(P^2N)$; the MAD adds no asymptotic term, and in practice the pool winnows fast.

Everything else is held fixed, deliberately, because the diagnosis blamed selection alone. `evolve_one_generation` builds the per-case absolute-error matrix once per generation by evaluating every tree on the training inputs and passes it into `selection`, which slices it to the current candidate set at each gate; if the matrix is ever absent, `selection` falls back to a size-7 tournament so it is safe to call either way. Crossover and mutation stay the standard uniform-point subtree operators from the previous rung, and `fitness_function` stays raw MSE, because the outer loop still tracks the best-ever tree by it for elitism and reporting and still carries the raw-MSE best forward unconditionally.

The falsifiable bet against the standard-GP numbers: spreading selection pressure across cases should kill the premature convergence that produced the worst outcomes. The cleanest prediction is that Nguyen-7's $-1.0$ seed disappears — case-by-case filtering keeps alternative forms alive so no single early winner locks the population — with all three seeds positive and the mean climbing from 0.330 toward the $\sim$0.97 the two good seeds already showed. On Nguyen-10, lexicase should *preserve* the $\sin(x)$ and $\cos(y)$ specialists that tournament buried, lifting the mean above 0.588. Koza-3 is the control: already solved, it should stay near 0.99. If instead Nguyen-10 does not improve, the bottleneck there is not selection diversity but raw representational reach, and the lever moves off selection. The one cost to watch honestly is wall-clock: $O(P^2N)$ selection is far heavier than tournament's $O(Pk)$, and the question is whether that compute buys back the seeds standard GP threw away.

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
