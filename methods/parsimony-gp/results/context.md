# Context: controlling program growth in tree-based genetic programming (circa 2007-2008)

## Research question

Genetic programming (GP) evolves a population of computer programs — for symbolic
regression, mathematical expression trees over a primitive set of functions and terminals —
by repeatedly selecting fitter individuals and recombining them with subtree crossover and
mutation. Run such a system for a few dozen generations and a robust, by-then-decade-old
phenomenon appears: after an initial phase in which the mean program size (node count) is
roughly static, the average size starts to climb, generation after generation, *without any
matching gain in fitness*. This is **bloat**. It is not the benign growth a problem may
genuinely require (a regression target may need a larger expression to fit all the cases);
bloat is growth that buys nothing in fitness. It is costly on every axis a practitioner cares
about: bloated programs are slow to evaluate (and slow to use afterwards), opaque to read,
and tend to generalise poorly.

The practical problem is sharp. The simplest and by far the most widely used remedy is to
shrink the fitness of a program by an amount that grows with its size, so selection prefers
smaller programs at equal quality. The intensity of that pressure is governed by a single
scalar, and the value of that scalar is the whole ballgame: set it too small and the
population still bloats wildly; set it too large and GP treats *minimising size* as its real
objective, almost ignoring fitness, and converges on extremely small but useless programs.
Worse, the "right" value is not portable — it depends on the problem, the primitive set, the
selection scheme, the population size, and it *drifts as the run proceeds*. There is, at this
point, essentially no theory to set it; users proceed by trial and error, and even a
hand-tuned constant can only ever achieve partial control over how the mean size evolves over
time. The goal a real solution must meet: a *principled, computable* prescription for that
scalar — one that can be set automatically, generation by generation, from quantities already
in hand, and that delivers tight, predictable control over the average program size across
problems and selection schemes, ideally even letting the practitioner *dictate* the size
trajectory rather than merely cap it.

## Background

**The GP substrate.** A tree-based GP individual is an expression tree (equivalently, a
prefix-notation program) over a function set (e.g. `+ - * /`, `sin`, `log`) and a terminal
set (input variables, constants). The initial population is generated randomly: the *full*
method picks functions until a depth limit and then terminals, giving bushy trees; the *grow*
method picks freely from functions and terminals until the limit, giving varied shapes;
Koza's *ramped half-and-half* combines them across a range of depth limits for size/shape
diversity. The dominant selection mechanism is **tournament selection** — draw `k` individuals
at random, keep the best — which automatically rescales fitness (it cares only about rank, not
magnitude) and keeps selection pressure roughly constant; the older alternative is
**fitness-proportionate (roulette-wheel) selection**, where an individual's selection
probability is its fitness divided by the total fitness. The workhorse operators are **subtree
crossover** (pick a node in each parent, swap the subtrees rooted there) and **subtree
mutation** (replace a randomly chosen subtree with a freshly generated random subtree). Crossover
points are usually biased 90% toward internal function nodes and 10% toward leaves, since
uniform choice would mostly swap single leaves. A small number of elite individuals are often
copied unchanged into the next generation. Hard depth or size limits (a common default depth
cap is 17) are typically enforced by rejecting an over-limit offspring and returning a parent.

**Observed dynamics — the diagnostic facts about existing systems.** Bloat has been studied
hard, and several phenomena about *existing* GP runs are established and load-bearing:

- *It is real and near-universal.* Across problems, mean size eventually grows rapidly with
  flat fitness; on landscapes whose fitness is essentially a function of size, growth is
  especially fierce.
- *Three classic qualitative theories.* The **replication-accuracy** theory (McPhee & Miller
  1995): success favours offspring functionally similar to the parent, pushing toward
  representations that protect fitness — which tend to be bloated. The **removal-bias** theory
  (Soule & Foster 1998): inactive code sits low in the tree (small subtrees); crossover that
  excises an inactive subtree keeps fitness but on average inserts a larger subtree, so
  offspring grow while retaining parental fitness. The **nature-of-search-spaces** theory
  (Langdon & Poli 1997): above some size the distribution of fitnesses stops varying with
  size, and since there are combinatorially more large programs than small ones of any given
  fitness, GP drifts toward longer programs simply because they are more numerous.
- *A quantitative constraint — the size evolution equation* (Poli 2001; Poli & McPhee 2003).
  An exact result for any GP with selection and symmetric subtree crossover: the mean size at
  the next generation equals the size-weighted *selection* probabilities,
  `E[μ(t+1)] = Σ_ℓ ℓ · p(ℓ,t)`, where `p(ℓ,t)` is the probability that selection picks a
  program of size `ℓ`. Equivalently the expected *change* is
  `E[μ(t+1) − μ(t)] = Σ_ℓ ℓ · (p(ℓ,t) − Φ(ℓ,t))`, with `Φ(ℓ,t)` the proportion of size-`ℓ`
  programs currently in the population. This does not by itself *explain* bloat, but it
  constrains it: for symmetric crossover, mean size evolves *as if selection alone acted*, so
  any change in mean size must come from selection treating some length classes differently
  from their population proportions — concretely, bloat requires that some shorter-than-average
  programs are selected below their proportion and/or some longer-than-average programs above
  it.
- *The crossover-bias theory* (Dignum & Poli 2007), consistent with that equation: subtree
  crossover removes as much material as it inserts on average, so it does not change mean size,
  but it does reshape the size *distribution*, driving the population toward a characteristic
  distribution (a Lagrange distribution of the second kind) heavy in very small programs.
  Since tiny programs essentially never solve a real problem, selection systematically
  discards the small ones it manufactures and keeps the larger, so mean size creeps up.

**A neighbouring identity.** Price's theorem (Price 1970) is a general result in evolutionary
dynamics: the one-generation change in the population mean of any heritable feature equals the
covariance, across the population, between that feature and fitness, divided by mean fitness
(plus a transmission term). It has been applied widely to gene frequencies; whether *program
size* qualifies as such a feature had, to this point, only been argued informally.

## Baselines

**Standard GP with a constant parsimony coefficient (Koza, *Genetic Programming*, MIT Press
1992).** The canonical GP setup: ramped half-and-half initialisation, tournament (or
fitness-proportionate) selection, 90%-rate subtree crossover, low-rate subtree mutation, one
elite, depth cap 17. To fight bloat, replace the raw fitness `f(x)` of program `x` by a
size-penalised fitness `f_p(x) = f(x) − c · ℓ(x)`, where `ℓ(x)` is the program's size and `c`
the **parsimony coefficient**, held *constant* through the run (the original `f` is still used
to recognise solutions and stop). Bigger programs lose more fitness and have fewer children.
This is the simplest, most widely deployed bloat control, and connects naturally to the
generalisation-versus-accuracy tradeoff and to Minimum-Description-Length penalties.
**Limitation:** the only general way to choose `c` is trial and error, and the good value is
problem-, primitive-, and selection-dependent; moreover, because the population statistics
that determine how much pressure a given `c` exerts change every generation, a fixed `c` can
only ever achieve *partial* control over the size trajectory — it cannot hold mean size on a
chosen course, and the same `c` that is mild early can be punishing late, or vice versa.

**Experimentally-adapted coefficient (Zhang & Mühlenbein, *Evolutionary Computation* 3(1),
1995).** Recognising that a constant is too rigid, this line lets `c` be re-adjusted at each
generation, and reported benefits from doing so (on evolving Sigma-Pi neural networks with
GP). **Limitation:** the adjustment is empirical/heuristic — there is no derivation that says
*what* value `c` should take at generation `t` to achieve a stated effect on mean size, so it
remains a tuned schedule rather than a computed one, and most implementations in the
literature fall back to a constant anyway.

**Anti-bloat operators (size-fair crossover/mutation; Langdon 2000; Crawford-Marks & Spector
2002; hoist and shrink mutation).** Instead of touching fitness, constrain the operators so
they cannot grow programs: size-fair crossover bounds the inserted subtree by the size of the
excised one; hoist mutation replaces a subtree with a subtree *of itself* (always smaller);
shrink mutation replaces a subtree with a single terminal. **Limitation:** these bias or
restrict the search operators, which can curtail exploration, and they act on *individual*
operations rather than giving any handle on the *population-level* mean-size dynamics; they do
not let you state and hit a target size.

**Hard size/depth limits (Koza 1992).** Reject any offspring exceeding a size or depth cap and
return a parent. **Limitation:** parents that nearly violate the cap get copied more often, so
the population fills up with programs jammed against the limit (stringy under size caps, bushy
under depth caps) — control by truncation, not by shaping the dynamics, and again no ability
to follow a chosen size trajectory.

**Tarpeian method (Poli 2003).** Act directly on the selection probabilities of the size
evolution equation: with some frequency, set the fitness of a randomly chosen
longer-than-average program to zero so it cannot be a parent (and need not even be evaluated,
saving time). Tuning that frequency modulates the anti-bloat intensity. **Limitation:** the
frequency is again a hand-set knob with no closed-form link to a desired mean-size outcome.

**Multi-objective bloat control (Ekart & Németh 2001; Kotanchek et al. 2006).** Treat fitness
and size as two objectives and select by a Pareto criterion. **Limitation:** it changes the
problem into a multi-objective search (with its own diversity and selection complications) and
still offers no single computable dial for the mean-size trajectory.

## Evaluation settings

The natural yardsticks already in use for measuring size control and solution quality:

- **Problems engineered to bloat.** A *Hole* problem assigning fitness 0.001 to programs
  under 10 nodes and 1.0 to all others (minimal conditions for bloat under crossover-bias),
  and a *Square Root* problem with fitness `f(x) = sqrt(ℓ(x))` (the whole landscape rewards
  size — a stress test). For both, fitness depends only on size, so any primitive set gives
  the same behaviour.
- **Classical symbolic-regression benchmarks.** Polynomials `x + x^2 + ... + x^d` sampled at
  21 equally spaced points `x ∈ [−1, 1]` (degrees `d = 6`, `d = 8`: *Poly-6*, *Poly-8*), with
  fitness `1/(1 + error)` and `error` the summed absolute deviation over the fitness cases.
  Other standard symbolic-regression targets in this family include univariate transcendental
  functions (e.g. `log(x+1) + log(x^2+1)` on `[0,2]`), bivariate trigonometric functions
  (e.g. `2·sin(x)·cos(y)` on `[−1,1]^2`), and even-degree polynomials such as
  `x^6 − 2x^4 + x^2` on `[−1,1]`.
- **A Boolean benchmark.** The 6-multiplexer (6 inputs, 2 address + 4 data lines, 64 fitness
  cases), fitness = number of cases correctly computed.
- **Protocol.** Multiple GP systems (two linear register-based, one tree-based), populations
  of 100 / 1,000 / 2,000 / 10,000, runs of 100-500 generations, averaged over many (35-100)
  independent runs with error bars; crossover at 100% in the linear systems; selection by
  fitness-proportionate in one system and binary tournament in another, precisely to test
  whether a size-control scheme transfers across selection schemes. The quantities tracked are
  the *mean program size per generation* (the size-control metric) and, separately, the
  success rate / solution accuracy. For symbolic regression more broadly, generalisation is
  read off via R^2 / RMSE on a held-out test set.

## Code framework

The size-control mechanism plugs into a GP harness that already exists. A program is an
expression tree with the usual operations; the data pipeline, the random-tree generator, the
fitness evaluation, tournament selection, and the subtree operators are all in place. What is
*not* settled is how the per-generation loop should shape selection so that average program
size stays under control without sacrificing fitness — that mechanism is the empty slot.

```python
import random
import numpy as np


# --- primitives that already exist in the GP toolkit ---
# safe_evaluate(tree, X)                  -> np.ndarray of predictions (NaN/inf guarded)
# generate_tree(method, max_depth, n_features) -> Tree   (method = 'grow' | 'full')
# Tree.copy()           -> deep copy
# Tree.size()           -> number of nodes
# Tree.depth()          -> tree depth
# Tree.get_all_nodes()  -> list of (node, parent, child_index)


def fitness_function(tree, X, y):
    """Raw error of a candidate expression — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Standard tournament selection on whatever fitness vector it is handed.
    Returns n_select copies of selected individuals (lower fitness wins)."""
    selected = []
    pop_size = len(population)
    for _ in range(n_select):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Standard subtree crossover: graft a random subtree of parent2 into parent1."""
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
    """Standard subtree mutation: replace a random subtree with a fresh random one."""
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
    """Produce the next generation.

    The genetic operators and selection above are generic. The open question is
    how to keep the average program size in check across generations while still
    driving fitness down.
    """
    new_population = []

    # Elitism: carry the best individual (by true fitness) forward unchanged.
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    # TODO: choose the parent-selection signal for this generation, then fill
    #       the population with offspring via crossover / mutation / reproduction.
    while len(new_population) < pop_size:
        pass

    return new_population[:pop_size]
```
