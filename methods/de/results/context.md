# Context: direct-search global optimization over continuous spaces (early-to-mid 1990s)

## Research question

Across science and engineering — circuit and filter design, finite-element simulation, control,
signal processing — the recurring task is to choose a vector of real-valued parameters
`x = (x_1, ..., x_D)` that minimizes a scalar cost `f(x)`, where the constraints and competing
objectives `g_m` have already been folded into a single cost (a weighted sum `z = Σ_m w_m f_m(x)`
or a min-max `z = max_m w_m f_m(x)`). The cost is the hard kind: **nonlinear, non-differentiable,
and multimodal**, with no usable gradient — sometimes it is not even a formula but the read-out of
a physical experiment or a simulation that takes minutes to hours per evaluation. So the only
admissible tools are *direct search* methods that probe `f` by sampling points and comparing
values.

A method that practitioners would actually adopt has to meet four demands at once:

1. **Cope with non-differentiable, nonlinear, multimodal cost** — find the *global* minimum, not
   just the nearest local basin, regardless of the starting point.
2. **Parallelize**, because when one evaluation costs minutes the only way to a usable answer in
   reasonable wall-clock time is to spread evaluations across many processors.
3. **Few, robust control variables** — the method's own knobs must be small in number and easy to
   choose, ideally not re-tuned for every new problem.
4. **Reliable convergence** — the global optimum reached consistently across independent random
   restarts, not just occasionally.

Every existing direct-search method below meets some of these but pays for the others. The
unsolved crux, the one that makes "robust and easy to choose" so hard, is this: a direct-search
method works by adding a *random variation* to a parameter vector and keeping it if it helps, and
the **scale (and orientation) of that variation is what decides everything** — too large and the
search never settles, too small and it crawls or sticks in a local basin, and the right scale is
different for every problem and *changes as the search progresses* (large while exploring, small
while refining). Getting that scale right has, up to now, always required an external mechanism
with its own parameters to tune. Closing that gap — without adding yet more knobs — is the problem.

## Background

The field state. Direct-search optimization of continuous black-box functions rests on a small set
of well-established families, all sharing one skeleton: maintain one or more candidate vectors,
**generate a variation** of a vector, then **decide whether to accept** it. Most basic methods use
the *greedy* acceptance criterion — keep the new vector if and only if it lowers the cost. Greedy
acceptance converges quickly but is liable to get **trapped in a local minimum**, a well-known
failure mode. Two structural ideas exist to forestall that trapping. The first is to run **several
vectors in parallel**: a population carries diversity, and a vector stuck in a poor basin can be
rescued by information from better-placed members — this is the safeguard built into genetic and
evolutionary algorithms. The second is to **occasionally accept uphill moves** with a probability
that shrinks over time, the relaxation of greed used by simulated annealing; as the schedule
cools, the acceptance rule reverts to greedy.

The pivotal sub-problem in this skeleton is the *generation* step — how the variation of a vector
is produced — and specifically the **distribution of the perturbation that is added**. The
prevailing wisdom by this time is that the perturbation should be drawn from a parametric
distribution (usually Gaussian) whose **scale parameter must itself be controlled**, because a
fixed scale is wrong: it has to be large during exploration and small during refinement, and it
must match the geometry of the particular cost surface. The amount of machinery the field has
built around *controlling that scale* — success-rule schedules, self-adapting strategy parameters,
cooling schedules — is itself testimony to how central, and how stubborn, the scale-setting problem
is.

A second, quieter idea is already in the air, in the simplex method: rather than perturb with an
externally specified distribution, **read the geometry of the move off the set of points you are
already carrying**. A simplex of `D+1` vertices expands and contracts relative to its own current
spread, with no user-set scale — a self-organizing move whose size tracks the landscape. That this
is possible *in principle* — extracting the scale of the next move from the configuration of the
current points — is a known and attractive property, even though the simplex realizes it only for a
single `D+1`-vertex figure and only as a local search.

Diagnostic facts about these existing systems:

- Greedy direct search alone converges fast but misconverges to local minima on multimodal cost.
- Population/parallel methods resist local-minimum trapping but, on continuous problems, their
  variation step is a fixed-distribution mutation that does not track the search's progress.
- Annealing can climb out of local minima but is **slow** (very large numbers of function
  evaluations) and is governed by **many** control variables (an adaptive-simulated-annealing
  implementation exposes on the order of a dozen, of which a couple dominate behavior).
- Methods whose perturbation scale is set or adapted by an explicit mechanism succeed only when
  that mechanism is itself tuned correctly; the adaptation is an extra moving part that can
  mistrack the true geometry.

## Baselines

The prior methods a new continuous global optimizer would be measured against and would react to.

**Greedy direct search — Nelder–Mead simplex (Nelder & Mead 1965; Bunday et al. 1987).** A
self-organizing local minimizer. For a `D`-parameter cost it keeps a polyhedron of `D+1` vertices,
each a parameter vector. New vertices are produced by **reflecting** the worst vertex through the
centroid of the rest, and by **expanding** or **contracting** the figure, the size of each move
scaled by the simplex's own current extent; the new vertex replaces its predecessor if it has lower
cost. This lets the search figure expand and contract to fit the landscape with no user-set scale.
*Gap:* it is fundamentally a **local** minimizer — a single simplex with no population diversity —
and, as a global search even with annealing bolted on, not powerful enough; once the figure
collapses into a basin it cannot leave it.

**Evolution strategies (Rechenberg 1973; Schwefel 1995).** Population methods that mutate a parent
by adding a **normally distributed** perturbation, coordinate-wise `x_j ← x_j + N(0, σ²)`. The
spread `σ` is a *strategy parameter* that must be controlled, and the family is largely defined by
*how* it is controlled: Rechenberg's **1/5 success rule** adjusts `σ` so that on average about one
mutation in five succeeds (derived for the sphere and the inclined-ridge models), and later
variants **self-adapt** a per-coordinate `σ` (or a full covariance) by evolving the strategy
parameters alongside the object parameters. *Gap:* the perturbation is a **predetermined parametric
distribution whose scale and orientation are set by a separate control mechanism** — a success-rule
schedule or an evolving `σ`/covariance — which adds control variables and is itself an adaptation
that can mistrack the cost surface, especially when the landscape's correlation structure differs
from what the strategy parameters have learned.

**Genetic algorithms (Holland; Goldberg 1989).** Population-based search with selection, crossover,
and mutation. Selection (e.g. tournament) biases reproduction toward fitter members; crossover
recombines two parents; mutation perturbs genes. On continuous problems real-coded variants are
used — simulated binary crossover blends two parents around their midpoint with a spread set by a
distribution index, and polynomial mutation perturbs a gene by a polynomially distributed amount
within bounds — each parameterized by a fixed distribution index and a per-gene mutation
probability (commonly `1/D`). *Gap:* the mutation that supplies fresh variation draws from a
**fixed distribution whose width is a user-set constant**, not coupled to how spread-out the
population currently is; the variation scale neither shrinks as the population converges nor aligns
with the shape of the basin being explored, so it is right for at most one phase of one problem.

**Simulated annealing (incl. Adaptive Simulated Annealing, Ingber 1992/1993).** Relax the greedy
rule: accept an uphill move of size `Δ` with probability `≈ exp(−Δ / T)`, cooling the temperature
`T` over time so the rule reverts to greedy. Adaptive variants adjust the cooling schedule and the
move-generation distribution on the fly. *Gap:* it needs **many control variables** (an adaptive
implementation exposes roughly a dozen, with a couple — the temperature-ratio and
temperature-anneal scales — dominating) and pays for its global reach with **large numbers of
function evaluations**, an expensive proposition when each evaluation is a slow simulation.

The common limitation across all four — the one a new method would have to get past — is that the
random variation added to a vector has a scale (and, for elongated or curved valleys, an
orientation) fixed by an external device: a single shrinking simplex, a controlled or self-adapted
`σ`/covariance, a constant mutation width, or a cooling schedule. None of these supplies a simple,
problem-robust way for the variation to stay appropriate as the search moves from broad
exploration to local refinement, and that extra tuning or adaptation machinery is exactly what
makes the methods hard to set and brittle across problems and across the phases of a single run.

## Evaluation settings

The natural yardstick already in use is a testbed of standard continuous minimization functions,
all with known global minima, run for a fixed dimensionality and box-constrained domain, with the
metric being how reliably and how *cheaply* (in number of cost-function evaluations, nfe, and in
generations to converge) a method reaches the global minimum across independent trials.

- **The De Jong suite**: the sphere `f(x) = Σ x_j²` (an easy, smooth, unimodal baseline); the
  two-parameter **Rosenbrock** saddle `100·(x_2 − x_1²)² + (1 − x_1)²` (a narrow, curved valley
  that defeats axis-aligned moves); the **step** function (plateaus); and a **noisy quartic**.
- **Highly multimodal functions**: **Rastrigin** `A·n + Σ (x_j² − A·cos(2π x_j))` with
  `A = 10` (a regular lattice of many local minima over `[−5.12, 5.12]`); **Ackley**
  `−20·exp(−0.2·√(mean x_j²)) − exp(mean cos(2π x_j)) + 20 + e` over `[−32.768, 32.768]`;
  **Griewank** (product-of-cosines ripples on a quadratic bowl); Shekel's foxholes; Corana's
  parabola (a riddled paraboloid).
- **Constrained / scaled problems**: Zimmermann's corner-constrained problem; Chebyshev-polynomial
  coefficient fitting, which forces parameters of grossly different magnitudes.

The dimensionalities span low (`D = 2` for Rosenbrock) up to high (`D = 100`), with the search box
fixed per function (e.g. Rastrigin on `[−5.12, 5.12]`, Ackley on `[−32.768, 32.768]`). The
protocol is to tune each method's control variables to its best per-function setting and report the
averaged nfe to reach the global minimum (a hyphen denoting misconvergence), and the generation at
which a run reaches close to its final value, across repeated independent runs. The competing
direct searches set up as references are an **annealed Nelder–Mead** and **adaptive simulated
annealing**.

## Code framework

A continuous black-box optimizer plugs into a generic population-based evolutionary harness that
already exists. The harness owns the things that are not in question: a representation of an
individual as a fixed-length vector of real genes with an attached scalar fitness; a benchmark cost
function that maps an individual to its cost; bounds `[lo, hi]` per coordinate and a clip; random
initialization of a population over the box; and per-generation bookkeeping that tracks the best
cost. The unsettled part is the generation step as a whole: given the current population, produce
the next population. At this stage that slot is intentionally neutral; it does not assume how many
members are consulted, how coordinates are changed, or how replacement is decided.

```python
import random
from typing import Callable, Tuple

from deap import base, creator, tools

# --- already exists: individual = list of floats with a scalar fitness ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))   # single-objective minimization
creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual(toolbox, dim: int, lo: float, hi: float):
    """Random individual uniformly over the box [lo, hi]^dim."""
    return creator.Individual([random.uniform(lo, hi) for _ in range(dim)])


def clip_individual(individual, lo: float, hi: float):
    """Keep genes inside the box."""
    for i in range(len(individual)):
        individual[i] = max(lo, min(hi, individual[i]))
    return individual


def make_generation(population: list, toolbox, dim: int, lo: float, hi: float) -> list:
    # TODO: the rule we will design here
    pass


def run_evolution(evaluate_func: Callable, dim: int, lo: float, hi: float,
                  pop_size: int, n_generations: int,
                  cx_prob: float, mut_prob: float, seed: int) -> Tuple[list, list]:
    """Evolve the population; return (best_individual, fitness_history)."""
    random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    fitness_history = []
    for gen in range(n_generations):
        pop = make_generation(pop, toolbox, dim, lo, hi)

        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```

The harness draws and evaluates the initial population and reports the best; the single
per-generation population-update rule is the slot to be filled.
