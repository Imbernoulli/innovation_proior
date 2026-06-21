## Research question

Symbolic regression searches the space of mathematical expressions for one that fits observed `(X, y)` data — not a black-box predictor, but an explicit, readable formula like `log(x+1) + log(x²+1)` or `2·sin(x)·cos(y)`. The object being designed is the **search strategy itself**: the fitness it selects on, how parents are chosen, how crossover and mutation build offspring, elitism, parsimony pressure, and diversity maintenance. The surrounding substrate — the tree representation, the protected operators, the benchmark data, and the outer evolution loop — is fixed.

## Prior art / Background / Baselines

- **Polynomial / least-squares regression.** Fit a fixed template `y = a₀ + a₁x + a₂x² + …` by minimizing `Σ(y − ŷ)²` in closed form.
- **Genetic algorithm on fixed-length strings.** Maintain a population of fixed-length chromosomes and evolve them by fitness-proportional selection, one-point crossover, and mutation.
- **Evolving variable-size structures.** Apply evolutionary operators to program-like objects rather than flat strings, allowing representational complexity to grow.

## Fixed substrate / Code framework

A self-contained GP framework is frozen and must not be touched. It supplies:

- **Tree representation.** A `Node` with `value` and `children` (`__slots__ = ('value','children')`, so no per-node attributes can be added). Operators are real→real and **protected** so any subtree is legal in any slot: `protected_div` returns `1.0` on a near-zero divisor; `protected_log` is `log(|a|)` (0 near zero); `protected_exp` clips its exponent to `[−10, 10]`. The function set is `{add, sub, mul, div, sin, cos, log, exp}`; terminals are input variables `xi` and ephemeral random constants drawn from `[−5, 5]`.
- **Helpers a strategy may call.** `safe_evaluate(tree, X)` (runs the tree, sanitizes NaN/Inf, clips finite); `generate_tree('grow'|'full', max_depth, n_features)` (a fresh random tree); `Tree.copy()/.size()/.depth()/.get_all_nodes()` (the last returns `(node, parent, child_idx)` tuples in preorder, so a chosen subtree can be spliced).
- **Initialization & outer loop.** `ramped_half_and_half` seeds generation 0; the loop scores the whole population each generation, tracks the single best tree ever seen by **raw MSE**, calls `evolve_one_generation` for the next population, and after the last generation evaluates the best-ever tree on the held-out test set. `r2_score` floors negative R² (a blow-up) at 0.0 and returns the discovered expression string.

## Editable interface

Exactly one region of `gplearn/custom_sr.py` is editable — five functions that *are* the search strategy. Every method on the ladder is a fill of this same contract:

- `fitness_function(tree, X, y) -> float` — lower is better (selection minimizes it).
- `selection(population, fitnesses, n_select, ...) -> list` — return `n_select` selected individuals (copies).
- `crossover(parent1, parent2, n_features, max_depth=17)` — return one offspring tree.
- `mutation(parent, n_features, max_depth=17)` — return one mutated tree.
- `evolve_one_generation(population, fitnesses, X_train, y_train, n_features, pop_size, crossover_rate=0.9, mutation_rate=0.05, max_depth=17) -> list` — the per-generation breeding logic; returns the next population (length `pop_size`).

The signature carries no persistent per-individual state across generations — `evolve_one_generation` sees only the current trees and their fitnesses, and `Node`'s `__slots__` blocks attaching extra fields — so any strategy must recompute everything it needs from the population each generation.

The starting point is the scaffold **default**: uniform random parent selection, crossover that returns a copy of `parent1`, and mutation that returns a copy of the parent. Each method on the ladder replaces exactly these definitions.

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

- **Nguyen-7** — univariate transcendental, `log(x+1) + log(x²+1)`, train `x ∈ [0, 2]` (20 points), test on a wider grid `x ∈ [−0.5, 2.5]` (100 points). Tests assembling logarithms and extrapolation.
- **Nguyen-10** — bivariate trigonometric, `2·sin(x)·cos(y)`, `(x, y) ∈ [0, 2π]²` (100 train, 400 test). Tests multivariate composition of trig primitives.
- **Koza-3** — univariate polynomial, `x⁵ − 2x³ + x`, `x ∈ [−1, 1]` (20 train, 100 test). A form the search *can* represent exactly, testing whether it finds it.

Each is run over three seeds {42, 123, 456} with population 500, 50 generations, an initial depth limit of 6 and an offspring depth cap of 8. The reported metric is **R²** on the held-out test set (higher is better, max 1.0; negative blow-ups floored at 0.0). RMSE and the discovered expression are reported alongside as feedback.
