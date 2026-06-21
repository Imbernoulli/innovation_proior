# Context: parent selection for genetic-programming symbolic regression (circa 2015)

## Research question

In tree-based genetic programming (GP) for symbolic regression, a population of candidate
expressions is evolved by repeatedly choosing *parents* and recombining them. The single
choice that most directly steers where the search goes is the **parent selection** rule:
given each individual's behavior on a training set, decide which individuals get to breed.
The standard practice reduces an individual's behavior to one scalar — its aggregate error
over all training cases (mean absolute error, or mean squared error) — and selects on that
scalar. The question is what selection rule to use when each individual's full per-case error
vector is available as a richer signal.

Concretely: an individual `i`'s behavior on a training set `T = {(y_t, x_t)}_{t=1}^N` is an
*error vector* `e(i) = |y - ŷ(i, x)| ∈ ℝ^N`, one entry per case. Aggregation collapses this
vector to a single number `f(i) = (1/N) Σ_t e_t(i)`.

## Background

**Genetic programming for symbolic regression.** A symbolic-regression GP maintains a
population of expression trees over a terminal/function set (variables, ephemeral random
constants, and operators such as `+ − * / sin cos exp log`). Each generation it evaluates
every tree on the training data, selects parents by some rule, and produces the next
generation by **subtree crossover** (swap a random subtree of one parent for a random
subtree of another) and **subtree mutation** (replace a random subtree with a freshly
generated one), usually with a max-depth/size cap to control bloat and with elitism (carry
the best individual forward unchanged). The Koza-style harness fixes most of these knobs;
the contribution being sought lives entirely in the *selection* step.

**Aggregation and behavioral information.** Krawiec & Liskowski (2015) discuss the
relationship between the error vector `e(i)` and the scalar `f(i)` derived from it. Most
aggregation schemes treat every test case as equally informative and produce a single
ranking of individuals across the whole population.

**Modal and "uncompromising" problems.** A separate line of work observed that many target
problems are *modal* (Spector 2012): they require qualitatively different modes of response
on different input regions, so no single averaged competence describes success. A related
notion (Helmuth, Spector & Matheson 2015) is the *uncompromising* problem — one for which an
acceptable solution must perform as well as possible on *each* test case; no good performance
on some cases can compensate for poor performance on others.

**Diversity and premature convergence.** GP populations are prone to collapsing onto one
behavioral family, after which crossover mostly shuffles near-identical material and the
search stalls. Selection rules that spread pressure across many distinct behaviors tend to
maintain higher population diversity, which is empirically associated with continued progress.

**A robust dispersion statistic.** For a vector of values `v`, the standard deviation
`σ(v)` is the usual measure of spread but is highly sensitive to outliers — a single huge
value drags it up. The **median absolute deviation**,
`MAD(v) = median_j( |v_j − median_k(v_k)| )` (Pham-Gia & Hung 2001), measures the *typical*
deviation from the median and has a 50% breakdown point, so it is unmoved by a minority of
extreme values. GP populations are full of extreme-error individuals (newly generated junk,
divergent expressions), which makes the contrast between `σ` and `MAD` directly relevant to
anything computed over per-case errors across the population.

## Baselines

These are the parent-selection rules a new rule would be measured against and react to.

**Tournament selection (Koza-style).** Pick `k` individuals uniformly at random; return the
one with the lowest aggregate error. Time `O(PN)` to fill a generation (it reads the
precomputed scalar fitness of `k` individuals per event). Simple, cheap, and the default in
nearly every GP system.

**Implicit fitness sharing, IFS (binary) and non-binary NBIFS (Krawiec & Nawrocki 2013;
McKay 2001).** Reward solving cases that *few others* solve. Binary form, for the set `T_i`
of cases individual `i` solves and `n(t)` the number of individuals solving case `t`:
`f_IFS(i) = Σ_{t ∈ T_i} 1/n(t)` (maximize). Non-binary form, with raw per-case fitness
`f(i,t) ∈ [0,1]`: `f_NBIFS(i) = Σ_{t ∈ T} f(i,t) / Σ_{i'∈P} f(i',t)`. Time `O(PN)`.

**Historically-assessed hardness (Klein & Spector 2008).** Scale each case's error by the
population's success rate on that case, then aggregate — a way to up-weight hard cases.

**Co-solvability (Krawiec & Lichocki 2010).** Reward solving *pairs* of cases together:
`f_CS(i) = Σ_{t_j, t_k} 1/n(t_j, t_k)` where `n(t_j,t_k)` is the number of individuals solving
both. Time `O(PN²)`.

**Lexicase selection (Spector 2012; Helmuth, Spector & Matheson 2015).** Instead of
aggregating, select each parent by *filtering on cases one at a time in random order*. For a
single selection event: start with the whole population as the candidate pool; shuffle the
cases; take the first case, find the best (lowest) error on it among the current pool, and
**remove every individual that does not have exactly that best error**; if one individual
remains it is the parent; otherwise drop that case and repeat with the next; if cases run out,
pick a random survivor. Each selected parent is therefore elite on at least the first case
used, and because the case order is reshuffled every event, pressure shifts toward
individuals that are elite on whichever hard cases happen to come first — promoting
specialists and high diversity. Worst-case time `O(P²N)`, but in practice it uses only a few
cases per event and runs much faster than the worst case. It outperforms tournament, IFS and
co-solvability on discrete program-synthesis and Boolean problems, and on the "uncompromising"
problems it was designed for.

**Age-fitness Pareto survival, AFP (Schmidt & Lipson 2011).** Treat error and *age* (number
of generations since the oldest ancestor) as two objectives; breed randomly; keep survivors
by SPEA2-style environmental selection, injecting a fresh individual each generation as a
random restart. A strong modern symbolic-regression approach that works in a two-objective
Pareto frame.

**Multi-objective treatment of cases (NSGA-II, SPEA2, ParetoGP).** One could treat each test
case as an objective and run a many-objective optimizer over hundreds or thousands of cases.

## Evaluation settings

The yardsticks already in use for symbolic-regression GP:

- **Synthetic and real-world regression problems.** Real-world: Boston housing price
  estimation (Harrison & Rubinfeld 1978); the Tower problem (15-minute averaged distillation-
  tower time series, predict propylene concentration); a wind-turbine bending-moment
  identification problem; building energy-efficiency heating/cooling estimation (ENH, ENC;
  Tsanas & Xifara 2012). Synthetic: the UBall5D / Vladislavleva-4 function
  `y = 10 / (5 + Σ_{i=1}^5 (x_i − 3)²)`, drawn from a community benchmark suite (White et al.
  2012). Data split 70/30 into train/test (except benchmarks with a predefined test set),
  each split normalized to zero mean and unit variance and re-partitioned per trial.
- **Standard GP harness/settings.** Population size 1000; crossover/mutation split 80/20;
  program length limits `[3, 50]`; ephemeral-random-constant range `[−1, 1]`; generation
  limit 1000; 30 independent trials; terminal/function set
  `{x, ERC, +, −, *, /, sin, cos, exp, log}`; elitism = keep best. Every method uses subtree
  crossover and point/subtree mutation, plus an optional per-generation constant-perturbation
  hill-climbing step.
- **Metric and protocol.** Best-of-run mean absolute error on the held-out test set is the
  primary metric (lower is better); number of fitness cases actually used per selection event,
  wall-clock time, and population behavioral diversity (fraction of unique output vectors `ŷ`)
  are reported as diagnostics. Methods compared by median best test fitness, mean rank across
  problems, and a Friedman test.

## Code framework

The selection rule plugs into an existing tree-based GP harness. The data pipeline, the
expression-tree data structure, the evaluation primitive, and the generational loop already
exist. The tree API is fixed: `safe_evaluate(tree, X)` returns the tree's outputs on the rows
of `X`; `generate_tree(...)` makes a random tree; `Tree.copy()`, `.size()`, `.depth()`, and
`.get_all_nodes()` (returning `(node, parent, child_index)` triples) manipulate trees. `np`
and `random` are available. The exposed edit surface is the scalar fitness, parent
selection, subtree crossover, subtree mutation, and the one-generation assembly function.

```python
import numpy as np
import random


def fitness_function(tree, X, y):
    """Scalar fitness, lower is better."""
    # TODO: reduce the tree's behavior on (X, y) to one number.
    pass


def selection(population, fitnesses, n_select, tournament_size=7):
    """Return n_select selected parents (copies).

    What an individual's selection-relevant behavior is, and how to compare
    individuals to choose parents, is exactly the open question.
    """
    # TODO: the parent-selection rule we will design.
    pass


def crossover(parent1, parent2, n_features, max_depth=17):
    """Produce one child from two parents."""
    # TODO: use the tree operators already provided by the harness.
    pass


def mutation(parent, n_features, max_depth=17):
    """Produce one child from one parent."""
    # TODO: use the tree operators already provided by the harness.
    pass


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Build the next generation with elitism + crossover/mutation/reproduction."""
    new_population = []
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())
    # TODO: prepare any population-level behavior data needed by selection.
    while len(new_population) < pop_size:
        # TODO: select parents, apply one genetic operator, and append the child.
        child = None
        new_population.append(child)
    return new_population[:pop_size]
```
