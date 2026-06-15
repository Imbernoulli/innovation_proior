## Research question

Symbolic regression searches the space of mathematical expressions for one that fits observed
`(X, y)` data — not a black-box predictor that returns a number, but an explicit, readable formula
like `log(x+1) + log(x²+1)` or `2·sin(x)·cos(y)`. Genetic programming (GP) does this by maintaining a
population of expression trees and evolving them by selection, crossover, and mutation. The single
thing being designed here is the **search strategy itself**: the fitness it selects on, how parents
are chosen, how crossover and mutation build offspring, elitism, parsimony pressure, diversity
maintenance. Everything around the strategy — the tree representation, the protected operators, the
benchmark data, the outer evolution loop — is fixed. The central tensions are exploration vs.
exploitation, controlling expression complexity (bloat), and avoiding premature convergence onto a
local optimum that fits the sample but not the function.

## Prior art before the first rung (the search engines the first rung reacts to)

The first rung — a Koza-style standard GP — is itself the resolution of a line of search methods.
These are the ancestors it converged out of; the fixed substrate below is what they left behind.

- **Polynomial / least-squares regression.** Fix a template `y = a₀ + a₁x + a₂x² + …` and solve for
  the coefficients minimizing `Σ(y − ŷ)²` in closed form. Fast and optimal *for the chosen template*.
  Gap: the functional form is an input, not an output — if the data came from a logarithm or a product
  of sinusoids, no polynomial recovers it, and the search problem (which form?) is the whole point.
- **The genetic algorithm on fixed-length strings (Holland 1975; Goldberg 1989).** A
  representation-agnostic optimizer for combinatorial landscapes: a population of fixed-length
  chromosomes, fitness-proportionate selection, one-point crossover, mutation, run for generations.
  The selection-and-recombination engine is exactly right for a space with no gradient. Gap: the
  fixed-length string pins the solution's size in advance and shatters under crossover — cutting a flat
  string at a random byte slices through a nested subexpression and yields garbage.
- **Evolving variable-size structures (Cramer 1985; Fujiki & Dickinson 1987).** Early work applied
  evolutionary operators to program-like objects rather than flat strings, showing representational
  complexity could grow. Gap: left open a simple, general recipe for evolving symbolic formulas while
  keeping enough syntactic validity for the search to keep running.

The resolution the first rung uses: let the individual *be* the expression tree (variable-size,
hierarchical), and recombine by **subtree exchange** — because any subtree is itself a valid
expression, every swap yields a valid offspring. Holland's Darwinian loop, run directly on trees.

## The fixed substrate

A self-contained GP framework is frozen and must not be touched. It supplies:

- **Tree representation.** A `Node` with `value` and `children` (`__slots__ = ('value','children')`,
  so no per-node attributes can be added). Operators are real→real and **protected** so any subtree is
  legal in any slot: `protected_div` returns `1.0` on a near-zero divisor; `protected_log` is
  `log(|a|)` (0 near zero); `protected_exp` clips its exponent to `[−10, 10]`. The function set is
  `{add, sub, mul, div, sin, cos, log, exp}`; terminals are input variables `xi` and ephemeral random
  constants drawn from `[−5, 5]`.
- **Helpers a strategy may call.** `safe_evaluate(tree, X)` (runs the tree, sanitizes NaN/Inf, clips
  finite); `generate_tree('grow'|'full', max_depth, n_features)` (a fresh random tree);
  `Tree.copy()/.size()/.depth()/.get_all_nodes()` (the last returns `(node, parent, child_idx)` tuples
  in preorder, so a chosen subtree can be spliced).
- **Initialization & outer loop.** `ramped_half_and_half` seeds generation 0; the loop scores the
  whole population each generation, tracks the single best tree ever seen by **raw MSE**, calls
  `evolve_one_generation` for the next population, and after the last generation evaluates the
  best-ever tree on the held-out test set. `r2_score` floors negative R² (a blow-up) at 0.0 and returns
  the discovered expression string.

## The editable interface

Exactly one region of `gplearn/custom_sr.py` is editable — five functions that *are* the search
strategy. Every method on the ladder is a fill of this same contract:

- `fitness_function(tree, X, y) -> float` — lower is better (selection minimizes it).
- `selection(population, fitnesses, n_select, ...) -> list` — return `n_select` selected individuals
  (copies).
- `crossover(parent1, parent2, n_features, max_depth=17)` — return one offspring tree.
- `mutation(parent, n_features, max_depth=17)` — return one mutated tree.
- `evolve_one_generation(population, fitnesses, X_train, y_train, n_features, pop_size,
  crossover_rate=0.9, mutation_rate=0.05, max_depth=17) -> list` — the per-generation breeding logic;
  returns the next population (length `pop_size`).

The signature carries no persistent per-individual state across generations — `evolve_one_generation`
sees only the current trees and their fitnesses, and `Node`'s `__slots__` blocks attaching extra
fields — so any strategy must recompute everything it needs from the population each generation.

The starting point is the scaffold **default**, which is deliberately broken as a search: uniform
random parent selection (no fitness pressure), crossover that just returns a copy of `parent1`, and
mutation that returns a copy of the parent. The loop's elitism keeps generation 0's lucky best, but
nothing improves it. Each method on the ladder replaces exactly these definitions.

```python
# EDITABLE region of gplearn/custom_sr.py — default fill (no real search)
def fitness_function(tree, X, y):
    """Evaluate fitness of a candidate program. Lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select):
    """Default: pick individuals uniformly at random (no fitness pressure)."""
    selected = []
    for _ in range(n_select):
        idx = random.randint(0, len(population) - 1)
        selected.append(population[idx].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Default: no recombination — return a copy of parent1."""
    return parent1.copy()


def mutation(parent, n_features, max_depth=17):
    """Default: no perturbation — return a copy of the parent."""
    return parent.copy()


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Build the next generation: elitism, then breed via selection + the operators above."""
    new_population = []
    elite_idx = int(np.argmin(fitnesses))            # elitism: keep best individual
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        parents = selection(population, fitnesses, 2)
        r = random.random()
        if r < crossover_rate:
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```

## Evaluation settings

Three standard symbolic-regression benchmarks spanning the difficulty range:

- **Nguyen-7** — univariate transcendental, `log(x+1) + log(x²+1)`, train `x ∈ [0, 2]` (20 points),
  test on a wider grid `x ∈ [−0.5, 2.5]` (100 points). Tests assembling logarithms and extrapolation.
- **Nguyen-10** — bivariate trigonometric, `2·sin(x)·cos(y)`, `(x, y) ∈ [0, 2π]²` (100 train, 400
  test). Tests multivariate composition of trig primitives.
- **Koza-3** — univariate polynomial, `x⁵ − 2x³ + x`, `x ∈ [−1, 1]` (20 train, 100 test). A form the
  search *can* represent exactly, testing whether it finds it.

Each is run over three seeds {42, 123, 456} with population 500, 50 generations, an initial depth
limit of 6 and an offspring depth cap of 8. The reported metric is **R²** on the held-out test set
(higher is better, max 1.0; negative blow-ups floored at 0.0). RMSE and the discovered expression are
reported alongside as feedback.
