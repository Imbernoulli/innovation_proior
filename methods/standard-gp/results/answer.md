# Standard Genetic Programming for Symbolic Regression, distilled

Standard GP (Koza-style genetic programming) discovers a symbolic formula fitting data by
running Darwinian evolution directly on a population of **expression trees**. Each individual is
a parse tree over a function set (operators) and a terminal set (input variables and constants).
The population is bred forward by selection plus subtree crossover and subtree mutation, scored
by fit error, for many generations; the best tree ever seen is the discovered model. The key
move over the ordinary genetic algorithm is the representation: the candidate *is* the
variable-size expression tree, and recombination swaps subtrees — and since any subtree is a
valid expression, every recombination yields a valid offspring.

## Problem it solves

Symbolic regression: given a finite data sample `(X, y)`, find a function *in symbolic form*
that fits — recovering both the functional **form** and its numeric constants. Conventional
regression fixes the form (linear/polynomial template) and only solves for coefficients;
genetic programming searches the space of expressions of unrestricted size and shape, which has
no fixed parameter vector and no gradient, so it relies only on the ability to *score* a
candidate.

## Key idea

Represent each candidate as an **expression tree** over a function set `F` (e.g.
`+, −, ·, protected /, sin, cos, log, exp`) and terminal set `T` (input variables + constants).
Maintain a population; each generation, score every tree by fit error and build the next
generation by selecting parents (favoring fitter ones) and applying genetic operators:

- **Subtree crossover** (the core): copy two selected parents, pick a node in each, swap the
  subtrees rooted there. Because any subtree is itself a valid expression, every swap produces a
  syntactically valid, evaluable offspring (closure under recombination). This is the
  structure-preserving analogue of string crossover; useful subexpressions are the building
  blocks that recombination splices together.
- **Subtree mutation**: replace a random subtree with a freshly grown random subtree, injecting
  primitives/constants that crossover alone could never reintroduce once lost.
- **Reproduction**: copy a selected individual unchanged.

## Design choices and why

- **Closure = type consistency + evaluation safety.** Unrestricted subtree crossover puts any
  subtree in any slot, so every function must accept any input. All functions are real→real
  (type consistency), and arithmetic is **protected**: division returns `1.0` on a near-zero
  divisor; `log` uses `log(|a|)` and returns `0` near zero; `exp` clips its exponent (e.g. to
  `[−10, 10]`); outputs are sanitized of `NaN`/`Inf` and clipped finite. Without this, evolved
  trees poison runs with non-finite fitness.
- **Sufficiency.** `F` must be able to express the target — transcendental targets need
  `sin/cos/log/exp`; `{+, −, ·}` suffices only for polynomials.
- **Ephemeral random constants.** Real coefficients can't be enumerated in `T`; a special
  terminal draws a random constant (frozen per leaf) at tree-creation time. Crossover then moves
  and combines these constants, so coefficients are *evolved* by selection, not solved.
- **Ramped half-and-half initialization.** Build the initial trees over a *ramp* of depth limits,
  half by `full` (bushy, uniform-depth) and half by `grow` (irregular). `full` alone gives poor
  shape variety; `grow` alone has a size distribution highly sensitive to the terminal:function
  ratio. The combination gives variety in both size and shape.
- **Tournament selection (size `k`, e.g. 7).** Pick `k` individuals at random with replacement,
  take the best.
  Only *rank* matters, so it is invariant to fitness scale/spread: no single super-fit tree can
  swamp the population early, and selection pressure is controlled by `k` rather than by score
  compression late — both failure modes of fitness-proportionate selection. `k` is the greediness knob;
  `O(k)` per draw, no global sum, trivially parallel.
- **90/10 function/terminal crossover-point weights.** Leaves outnumber internal nodes
  in a typical tree, so uniform point choice would mostly swap single terminals. Giving function
  nodes much larger selection weight makes crossover usually exchange a meaningful subexpression.
- **Operator rates.** Crossover ≈ 0.9 (the workhorse — recombines building blocks), subtree
  mutation small (≈ 0.01–0.05, disruptive — only to refresh diversity), reproduction the
  remainder.
- **Elitism = 1.** Stochastic operators can destroy the best; copy the single best forward
  unconditionally so the best-of-generation never worsens.
- **Max-depth cap (e.g. 17) on offspring.** Controls **bloat** (programs growing without fitness
  gain via inert introns); offspring exceeding the cap are rejected (a parent is returned). Init
  depth limit smaller (≈ 6) so generation 0 starts compact.
- **Fitness = mean squared error**, lower better — the least-squares fit objective; tournament
  needs only its ranking.

## Final algorithm

```
initialize population of M random trees (ramped half-and-half over F, T; ephemeral constants)
best <- none
repeat for G generations:
    for each tree: fitness <- MSE(tree(X), y)            # lower is better
    update best with the lowest-fitness tree seen so far
    next_pop <- [ copy of current best ]                 # elitism
    while |next_pop| < M:
        r ~ Uniform(0,1)
        if   r < p_crossover:  child <- subtree_crossover( tournament(), tournament() )
        elif r < p_crossover + p_mutation:  child <- subtree_mutation( tournament() )
        else:                  child <- reproduce( tournament() )      # copy
        reject child if depth(child) > max_depth (return a parent instead)
        next_pop.append(child)
    population <- next_pop
return best                                              # the discovered symbolic formula
```

where `tournament()` samples `k` individuals uniformly with replacement and returns a copy of the best.

## Working code

Filling the search-strategy slots of the harness; the tree object exposes
`.copy()/.depth()/.get_all_nodes()`, and `safe_evaluate` / `generate_random` /
protected operators are the fixed substrate.

```python
import random
import numpy as np


def _is_function_node(node):
    return bool(getattr(node, "children", ()))


def _choose_subtree_index(tree):
    """Choose a crossover point with function nodes weighted 0.9 and leaves 0.1."""
    nodes = tree.get_all_nodes()
    weights = [0.9 if _is_function_node(node) else 0.1
               for node, _parent, _child_idx in nodes]
    return random.choices(range(len(nodes)), weights=weights, k=1)[0]


def _replace_subtree(tree, point, replacement):
    """Replace the subtree at point in a copied tree, including replacement at the root."""
    _node, parent, child_idx = tree.get_all_nodes()[point]
    if parent is None:
        return replacement.copy()
    parent.children[child_idx] = replacement.copy()
    return tree


def fitness_function(tree, X, y):
    """MSE fitness over the fitness cases — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection: sample k contenders, keep the best (min MSE). Rank-only, so it is
    invariant to fitness scale and holds selection pressure constant."""
    selected = []
    pop_size = len(population)
    k = max(1, min(tournament_size, pop_size))
    for _ in range(n_select):
        candidates = [random.randrange(pop_size) for _ in range(k)]
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())     # copy: parents may be reused
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Subtree crossover: graft a subtree of parent2 into parent1. Any subtree is a valid
    expression, so the offspring is always valid (closure)."""
    offspring = parent1.copy()
    donor = parent2.copy()

    off_point = _choose_subtree_index(offspring)
    don_point = _choose_subtree_index(donor)
    donor_subtree = donor.get_all_nodes()[don_point][0]
    offspring = _replace_subtree(offspring, off_point, donor_subtree)

    if offspring.depth() > max_depth:                 # bloat brake
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    """Subtree mutation: replace a random subtree with a freshly grown random subtree —
    injects material crossover cannot reintroduce."""
    offspring = parent.copy()

    # Headless-chicken mutation: build a random donor and graft one of its subtrees.
    donor = generate_random('half and half', min(6, max_depth), n_features)
    donor_point = _choose_subtree_index(donor)
    new_subtree = donor.get_all_nodes()[donor_point][0]

    mut_point = _choose_subtree_index(offspring)
    offspring = _replace_subtree(offspring, mut_point, new_subtree)

    if offspring.depth() > max_depth:
        return parent.copy()
    return offspring


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """One generation: elitism, then breed by tournament selection + crossover / mutation /
    reproduction in the given proportions."""
    new_population = []

    elite_idx = int(np.argmin(fitnesses))            # elitism: best survives
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:                        # subtree crossover (~0.9)
            parents = selection(population, fitnesses, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:      # subtree mutation (~0.05)
            parents = selection(population, fitnesses, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:                                         # reproduction (remainder)
            parents = selection(population, fitnesses, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```

This task-contract version keeps the Koza-style core: MSE fitness, tournament selection,
subtree crossover, subtree mutation, reproduction, explicit elitism, and a hard depth cap.
gplearn's `SymbolicRegressor` uses the same core representation/operator mechanics in
prefix-program form: `_tournament` samples contenders with replacement, `get_subtree` weights
function nodes by `0.9` and terminals by `0.1`, `crossover` splices prefix slices,
`subtree_mutation` uses the headless-chicken method, and bloat is controlled by parsimony and
additional hoist/point mutation options rather than this scaffold's explicit depth rejection.
