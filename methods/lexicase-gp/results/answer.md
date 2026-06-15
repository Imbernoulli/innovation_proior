# ε-lexicase selection, distilled

ε-lexicase selection is a parent-selection rule for genetic-programming symbolic regression.
It selects each parent by filtering the population on training cases one at a time, in a
fresh random order. At each case, candidates survive only if their error is within a
data-adaptive near-elite band: the best error `e*_t` plus a tolerance `ε_t` given by the
median absolute deviation of the per-case errors. It is the continuous-error generalization
of lexicase selection: where lexicase keeps only individuals *exactly* tied for best on each
case (which works on discrete errors but collapses to single-case selection on continuous
noisy regression errors), ε-lexicase keeps *near-elite* individuals, with `ε` set
automatically per case from the spread of the errors.

## Problem it solves

Aggregate fitness (MAE/MSE) collapses an individual's per-case error vector
`e(i) = |y − ŷ(i,x)| ∈ ℝ^N` into one scalar, discarding which regions it handles well and
penalizing specialists that are elite on a hard subset of cases but mediocre elsewhere —
exactly the individuals that carry the partial solutions GP must recombine. ε-lexicase
selects on the error *vector* without aggregating, rewarding individuals that are uniquely
good on subsets of cases (especially the hard, rarely-solved ones), and maintains high
behavioral diversity on continuous, noisy real-valued errors.

## Key idea

For one parent selection event:

1. Candidate pool ← entire population. (global pool)
2. Shuffle the training cases. (uniform random sequence)
3. While more than one candidate and cases remain: take the next case `t`; keep only
   candidates within `ε_t` of the best error on `t`.
4. Return the single survivor, or a random one if cases run out.

The pass condition relaxes lexicase's exact `e_t(i) = e*_t` to

```
e_t(i) ≤ e*_t + ε_t ,
```

with `e*_t` the best (minimum) error on case `t` and the threshold set by the **median
absolute deviation** of the per-case errors,

```
ε_t = λ(e_t) = median_j( | e_t(j) − median_k(e_t(k)) | ).
```

In the form landed on below, `e*_t` and `ε_t` are recomputed over the *current candidate
pool* at each gate (the pool-relative variant); see the variants section.

Why each piece:

- **Filter, don't aggregate.** A chain of "best on this case" gates selects individuals good
  on a *conjunction* of cases — a unique combination no sum can express.
- **Random case order per event.** Spreads selection pressure across all cases; a case's
  filtering strength is proportional to its difficulty, so hard cases landing early route
  selection to their specialists. Also drives high population diversity.
- **Relax the gate to `ε`.** On continuous noisy errors, exact ties are measure-zero, so
  plain lexicase's first gate empties the pool to one individual → single-case selection,
  worse than tournament. "Within `ε` of best" lets near-elites survive, so multiple cases
  participate and the conjunction mechanism is recovered.
- **`ε` = MAD, adaptive, not a user constant.** A fixed `ε` is problem-dependent and goes
  blind as the population converges (too wide late in a run → no filtering → random
  selection). MAD auto-scales `ε` to each case's error spread and shrinks as the population
  improves. MAD over the standard deviation `σ` because GP populations are full of
  extreme-error junk individuals: `σ` is dominated by those outliers (so `ε` would balloon
  and everyone passes), while MAD (50% breakdown point) reports the spread among the real
  contenders.
- **Anchor `ε` to the elite.** `e_t(i) ≤ e*_t + ε_t` makes the pass band relative to the best
  error on that case, so the elite always passes its own band and the pool is never emptied
  (when `e*_t` is the pool minimum). The selected parent is **ε-Pareto-optimal**: within the
  allowed ε band of the Pareto front over the cases, not necessarily on the exact front. An
  alternative anchors to the target `y` (`e_t(i) ≤ ε_t`, i.e. `|y_t − ŷ_t| ≤ ε_t`), but the
  elite-relative form preserves case-level selection pressure even when no program is close
  to the true target value. (With a frozen *population* elite, the static form needs an
  extra guard — if no current candidate passes a case, leave the pool unchanged — to stay
  nonempty; anchoring to the pool elite removes that need.)

## Threshold-reference variants

`e*_t` and/or `ε_t` can be computed over the whole population or over the current shrinking
candidate pool:

- **static** — both `e*_t` and `ε_t = λ(e_t)` computed once per generation over the whole
  population; selection is then discrete filtering on precomputed pass/fail flags. If no
  current candidate passes a case, that case leaves the pool unchanged. This is the
  canonical parameter-free form.
- **semi-dynamic** — `ε_t = λ(e_t)` over the population (once/gen), but the elite `e*_t`
  recomputed as the pool minimum at each gate. Closer to true lexicase; the bar rises toward
  the surviving contenders as the pool shrinks.
- **dynamic** — *both* the pool-elite `e*_t` and `ε_t = λ(e_t(S))` recomputed over the
  current pool `S` at each gate. As the pool homogenizes, the pool-MAD collapses, so `ε`
  shrinks and later gates keep filtering rather than waving everyone through.

All three share lexicase's `O(P²N)` worst-case complexity for selecting `P` parents over
`N` cases. The static preprocessing is `O(PN)`, and pool-level MAD computations in the
dynamic form are still linear in the current pool per case, so they add no asymptotic term.

## Final algorithm (one selection event, pool-relative form)

```
GetParent(population, errors):           # errors[i, t] = |y_t - y_hat_t(i)|
    pool  ← list of all individual indices
    cases ← shuffle(range(N))
    for t in cases:
        if |pool| ≤ 1: break
        e*    ← min_{i in pool} errors[i, t]                       # pool elite
        ε     ← median_{i in pool}(|errors[i,t] - median_{j in pool} errors[j,t]|)  # pool MAD
        pool  ← { i in pool : errors[i, t] ≤ e* + ε }             # relaxed elitist gate
    return random choice from pool
```

## Working code

Filling the exposed GP harness functions (`safe_evaluate`, `generate_tree`,
`Tree.copy/size/depth/get_all_nodes`, `np`, `random` are provided by the skeleton);
crossover, mutation, and elitism are standard tree-GP operators:

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
