In tree-based genetic programming for symbolic regression I evolve a population of expression trees, and every generation the single most consequential decision is which individuals get to breed. The default is tournament selection on aggregate error: I take each individual's vector of per-case errors $e(i) = |y - \hat y(i,x)| \in \mathbb{R}^N$, one entry per training point, crush it to a scalar like the mean absolute error $f(i) = \tfrac{1}{N}\sum_t e_t(i)$, and pick the lowest. The trouble is the crushing step. The whole premise of GP is that a full solution gets assembled out of partial solutions — subprograms that each get some region of the problem right, recombined by crossover until one tree gets all of it. But the instant I average $e(i)$ into a scalar I have deleted *which* regions an individual handles well. Two individuals can share an identical mean while one is mediocre everywhere and the other is the population's very best on a cluster of hard cases and poor on the rest. For harvesting a building block the specialist is gold and the generalist is nothing, yet aggregate selection cannot tell them apart, and it actively discards the specialist because its few weak cases inflate its average. The information I most need to steer the search — the shape of $e(i)$ — is exactly the information I throw away before I select.

What I want is pressure that rewards being uniquely good on *some* part of the problem, especially the hard cases few others solve, since those are where I am missing a piece. The weighting methods reach toward this but never escape aggregation. Implicit fitness sharing scores $f_{\mathrm{IFS}}(i) = \sum_{t \in T_i} 1/n(t)$, paying a lot for a case only a handful solve; historically-assessed hardness rescales each case's error by the population success rate; co-solvability rewards solving *pairs* together, $f_{\mathrm{CS}}(i) = \sum_{t_j,t_k} 1/n(t_j,t_k)$. Each is "reward the rare and hard," but each then *sums* the per-case rewards into one number before selecting. The combination is gone the moment the terms are added, so none of them can reward an individual for a unique *conjunction* of cases; co-solvability keeps only pairs, has no clean continuous version, and costs $O(PN^2)$. Aggregation is the disease and these treat symptoms.

I propose $\varepsilon$-lexicase selection. The structural move, inherited from lexicase selection, is to stop scoring-then-picking and instead *filter*. For one parent-selection event, start with the whole population as the candidate pool, shuffle the training cases into a fresh random order, then walk the cases as a sequence of gates: at each case keep only the candidates that are best on it, drop the rest, and stop when one candidate remains (or pick a random survivor if the cases run out). No averaging appears anywhere — just a chain of elitism filters — and an individual who survives a long chain is elite on case $A$ *and* elite-among-those on case $B$ and so on, i.e. good on a specific *conjunction* of cases, precisely the combination no sum can express. The random order is load-bearing: a fixed order would make the first case a permanent dictator and collapse to a deterministic lexicographic ranking, so I reshuffle for every event. Then over many events every case takes its turn at being first, pressure is spread across all cases, and a case's filtering strength is proportional to its difficulty — an easy case nearly everyone ties on barely shrinks the pool, a hard case where only one or two reach the best error slices the pool down to those specialists. Hard cases landing early route selection straight to the specialists I want to propagate, and because each parent comes down a different random filter path the population stays behaviorally diverse instead of collapsing onto one family.

The wall is what happens when I drop exact lexicase onto continuous, noisy regression errors. Its gate keeps individuals whose error *exactly equals* the pool-best error $e^*_t$ on the case. Two distinct expression trees essentially never produce the identical floating-point error on a case unless they are the same model, so the very first gate slams the pool from $P$ down to one, the parent is decided by a *single* noisy case, and the multi-case filtering that was the whole point is gone — single-case selection that is actually worse than tournament. The mechanism is sound; the pass condition is too sharp for a measure-zero event. So I relax the gate from "exactly best" to "near-best": individual $i$ passes case $t$ when

$$e_t(i) \le e^*_t + \varepsilon_t,$$

so a near-elite is no longer culled for losing by $10^{-7}$, a single case no longer empties the pool, multiple cases participate again, and the conjunction mechanism is recovered in continuous space.

The remaining question is what $\varepsilon_t$ should be, and a user-set constant is the wrong answer. A fixed band is problem-dependent — the same number gives sensible filtering on one dataset and almost none on another, since it depends on the error scale and the population spread — and it goes blind over a run: a band reasonable when errors are large early becomes, as the population converges and errors compress, so wide relative to the now-tiny spread that every survivor passes every case, the gates stop filtering, and selection turns random exactly when I should be fine-tuning. I want $\varepsilon$ to set itself from the data, tracking the *spread* of the per-case errors so that on a tightly-clustered case a small band separates the near-elites and on a contentious wide-spread case a larger band admits the contenders, shrinking automatically as the case gets solved. The textbook spread measure, the standard deviation $\sigma(e_t)$, fails immediately on GP populations because they are full of garbage individuals — freshly mutated junk, expressions that divide by near-zero and produce errors like $10^6$ — and $\sigma$ is dominated by those outliers, so the band would be set by the junk, balloon, and pass everyone. I therefore use the median absolute deviation,

$$\varepsilon_t = \lambda(e_t) = \operatorname{median}_j\!\big(\,\big|\,e_t(j) - \operatorname{median}_k(e_t(k))\,\big|\,\big),$$

whose inner median is the typical error and whose outer median is the typical distance from it. With a 50% breakdown point MAD simply does not see a minority of arbitrarily huge outliers, so it reports the spread among the real contenders — the scale at which "near-elite" should be judged — and it is parameter-free, auto-scaling to each case's difficulty and shrinking as the population converges. Anchoring the band to the elite as $e_t(i) \le e^*_t + \varepsilon_t$ rather than to the target $y$ (a $|y_t - \hat y_t| \le \varepsilon_t$ band) matters because the elite always clears its own band so the pool is never emptied, and case-level pressure survives even when no program is yet close to the true target value. With tolerance, selection no longer lands on the exact Pareto front over the cases but within the allowed slack of it: the selected parent is $\varepsilon$-Pareto-optimal, which is the honest weakening for noisy data — I refuse to treat a tiny continuous-error difference as meaningful, while still excluding anyone worse by more than the robust population scale.

One design axis remains: over which individuals do I compute $e^*_t$ and $\varepsilon_t$. Computing both once per generation over the whole population gives the *static* form, equivalent to converting the continuous errors into a population-level pass/fail table and running ordinary lexicase on it (with one guard — if no current candidate passes a case, that case leaves the pool unchanged, so a frozen population elite that has already been filtered out cannot empty the pool). Moving only the elite $e^*_t$ to the current pool minimum at each gate gives the *semi-dynamic* form, where the bar rises toward the survivors actually still in contention. Recomputing *both* the pool elite and the pool MAD $\varepsilon_t = \lambda(e_t(S))$ over the current pool $S$ at each gate gives the *dynamic* form: as the pool homogenizes the pool-MAD collapses, so $\varepsilon$ shrinks further and the later gates keep filtering hard instead of waving everyone through. I land on this pool-relative form for continuous regression because it is the one that keeps the gates discriminating all the way down the chain, never empties the pool (the pool-best always clears its own band), and slices the per-case error matrix I already have directly to the current candidate set. All three variants share lexicase's $O(P^2N)$ worst-case cost to select $P$ parents over $N$ cases — the pool minimum and MAD are linear in the current pool per case and add no asymptotic term — so the gain is not a better big-O but restoring genuine multi-case filtering on continuous errors with no hand-tuned tolerance, while in practice the pool winnows fast enough to keep wall-clock time in the tournament-selection ballpark.

Concretely, crossover and mutation stay as ordinary subtree operators, the scalar MSE fitness stays for elitism and as a tournament fallback, and the new ingredient is the per-case absolute-error matrix prepared once per generation, which the parent chooser slices to the current candidates at each gate to recompute the pool elite and the pool MAD:

```python
import numpy as np
import random


def fitness_function(tree, X, y):
    """MSE — lower is better. Used for elitism and as a selection fallback."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
    """epsilon-lexicase selection (pool-relative form).

    _errors has shape (P, N), with _errors[i, t] = |y_t - y_hat_t(i)|. At each
    gate the pool elite e*_t and tolerance epsilon_t = MAD(e_t) are recomputed
    over the current pool. Falls back to tournament if the matrix is absent.
    """
    selected = []
    pop_size = len(population)

    if _errors is None:                            # no behavioral matrix -> tournament
        for _ in range(n_select):
            candidates = random.sample(range(pop_size), min(7, pop_size))
            best = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best].copy())
        return selected

    n_cases = _errors.shape[1]
    for _ in range(n_select):
        candidates = list(range(pop_size))         # global pool: whole population
        cases = list(range(n_cases))
        random.shuffle(cases)                      # uniform random case sequence

        for case in cases:
            if len(candidates) <= 1:               # reduced to one -> selected
                break
            case_errors = _errors[candidates, case]
            min_err = float(np.min(case_errors))                       # pool elite e*_t
            mad = float(np.median(np.abs(                              # epsilon_t = MAD
                case_errors - float(np.median(case_errors)))))         #   over pool
            candidates = [c for c, e in zip(candidates, case_errors)   # relaxed gate
                          if e <= min_err + mad]

        winner = random.choice(candidates)         # tie -> random survivor
        selected.append(population[winner].copy())

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
    """Standard subtree mutation — replace a random subtree with a new random tree."""
    offspring = parent.copy()
    if offspring.size() <= 1:
        return generate_tree('grow', 3, n_features)
    mut_point = random.randint(1, offspring.size() - 1)
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
    """One generation with elitism + epsilon-lexicase selection."""
    new_population = []

    elite_idx = int(np.argmin(fitnesses))          # elitism: keep best
    new_population.append(population[elite_idx].copy())

    errors = np.empty((len(population), X_train.shape[0]))
    for i, tree in enumerate(population):
        y_pred = safe_evaluate(tree, X_train)
        errors[i] = np.abs(y_train - y_pred)

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2, _errors=errors)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1, _errors=errors)
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = selection(population, fitnesses, 1, _errors=errors)[0]
        new_population.append(child)

    return new_population[:pop_size]
```
