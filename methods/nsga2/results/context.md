## Research question

A problem with several conflicting objectives `f(x) = (f_1(x), ..., f_M(x))` (take all to be
minimized) does not have a single best solution. Improving one objective forces another to
get worse, so the right answer is a *set*: the Pareto-optimal set, the points no other
feasible point can beat on every objective at once. Formally `x` dominates `y` when
`f_m(x) <= f_m(y)` for all `m` and `f_m(x) < f_m(y)` for at least one `m`; the Pareto front is
the image of all non-dominated points. The practical goal is to recover a good approximation
of that entire front in **one** run, so a decision-maker can pick a trade-off afterward.

Because a population-based search carries many candidate solutions at once, an evolutionary
algorithm is the natural tool: a single run can, in principle, populate the whole front.
That gives two simultaneous objectives for the *algorithm itself*:

1. **Convergence** — drive the population down onto the true front.
2. **Diversity / spread** — keep the survivors spread out *across* the front, not bunched on
   one piece of it, and reaching its extremes.

The question is how to design an EA that pursues both objectives well.

## Background

By the late 1990s the dominant template for multiobjective evolutionary algorithms (MOEAs)
combined two ingredients introduced in the early-90s wave (Fonseca & Fleming's MOGA 1993;
Srinivas & Deb's NSGA 1994; Horn et al.'s NPGA 1994):

- **Fitness by non-domination ranking.** Goldberg (1989) proposed ranking a population by
  Pareto layers: find all non-dominated members (rank 1), set them aside, find the
  non-dominated members of the rest (rank 2), and so on. Selection then favors lower
  (better) ranks. This converts the vector objective into a single scalar rank that points
  the population toward the front.
- **Diversity by fitness sharing.** Goldberg & Richardson (1987) keep a population spread out
  by degrading an individual's fitness in proportion to how crowded its neighborhood is. With
  a distance `d(i,j)` between two individuals (in objective or parameter space), the sharing
  function is

  ```
  sh(d) = 1 - (d / sigma_share)^alpha   if d < sigma_share,   else 0     (alpha ~ 2)
  ```

  the niche count `m_i = sum_j sh(d_ij)` (counting self, `sh(0)=1`), and the shared fitness is
  `F_i / m_i`. Individuals in dense clusters get heavily discounted, so selection pushes the
  population to fan out.

Several empirical lessons had accumulated and frame the design space:

- **Elitism helps MOEAs.** Studies (Zitzler et al. 1999; Rudolph 1998) reported that carrying
  the best-so-far material forward measurably speeds convergence and reduces regressions where
  useful solutions are lost between generations.
- **Non-domination sorting cost.** Building the rank layers the naive way —
  recomputing pairwise dominations front by front — costs on the order of `O(M N^3)` for `M`
  objectives and population `N`.
- **`sigma_share` sensitivity.** The spread that fitness sharing achieves depends on
  the chosen `sigma_share`; it also presupposes a distance metric and an implicit guess about
  how many niches the front supports. Comparing every pair to compute niche counts is `O(N^2)`.

For real-coded (continuous-variable) GAs, two variation operators had become standard and
are taken as given building blocks:

- **Simulated binary crossover, SBX** (Deb & Agrawal 1995). A real-coded recombination that
  imitates the behavior of single-point crossover on binary strings. Define the spread factor
  `beta = |(c2 - c1) / (p2 - p1)|` (ratio of child spread to parent spread). Children are
  sampled from the polynomial density

  ```
  P(beta) = 0.5 (eta_c + 1) beta^{eta_c}            if beta <= 1
          = 0.5 (eta_c + 1) / beta^{eta_c + 2}      if beta > 1
  ```

  Drawing `u ~ U[0,1]` and inverting the CDF gives
  `beta_q = (2u)^{1/(eta_c+1)}` for `u <= 0.5`, else `(1 / (2(1-u)))^{1/(eta_c+1)}`, and the
  two children are
  `c1 = 0.5[(1+beta_q) p1 + (1-beta_q) p2]`, `c2 = 0.5[(1-beta_q) p1 + (1+beta_q) p2]`.
  The distribution index `eta_c` controls locality (large `eta_c` -> children near parents),
  and crucially the spread of children is proportional to the spread of the parents, so the
  operator self-adapts as the population contracts. A bound-respecting version recomputes the
  spread against each variable's distance to its `[x_L, x_U]` bounds so no child leaves the
  box.

- **Polynomial mutation** (Deb & Agrawal/Goyal). For a variable `p in [x_L, x_U]`, define
  the normalized distances to the bounds

  ```
  delta_L = (p - x_L) / (x_U - x_L),     delta_R = (x_U - p) / (x_U - x_L).
  ```

  Drawing `u ~ U[0,1]` and using `mut_pow = 1/(eta_m+1)`, the bounded polynomial step is

  ```
  if u < 0.5:
      xy = 1 - delta_L
      val = 2u + (1 - 2u) xy^{eta_m + 1}
      delta_q = val^{mut_pow} - 1
  else:
      xy = 1 - delta_R
      val = 2(1-u) + 2(u-0.5) xy^{eta_m + 1}
      delta_q = 1 - val^{mut_pow}

  p' = p + delta_q (x_U - x_L)
  ```

  The distribution index `eta_m` sets the perturbation scale `~ O((x_U - x_L)/eta_m)`; values
  in `[20, 100]` are typical, and a per-variable mutation probability `p_m = 1/n` (with `n`
  decision variables) mutates about one variable per individual on average.

## Baselines

**SPEA — Strength-Pareto EA (Zitzler & Thiele 1998/99).** Maintains an *external archive* of
all non-dominated solutions discovered so far, which participates in selection. Each archive
member's "strength" is the count of current-population solutions it dominates; a population
member's fitness is the sum of the strengths of the archive members that dominate it (lower is
better), which makes selection push toward the archive's non-dominated solutions. Diversity is
maintained by a deterministic clustering of the archive when it overflows. Implemented
naively it is `O(N^3)`; with careful bookkeeping `O(N^2)`.

**PAES — Pareto-Archived Evolution Strategy (Knowles & Corne 2000).** A `(1+1)`-ES: one parent
produces one mutated offspring, accepted or rejected by comparing dominance against the parent
and an archive of best solutions. When parent and offspring are mutually non-dominated, the
tie is broken by an *adaptive grid* that divides objective space into `2^{ld}` cells (depth `l`,
`d` variables) and prefers the offspring if it lands in a less crowded cell. Worst-case cost is
`O(a N)` for archive size `a`, hence `O(N^2)` overall.

**Rudolph's elitist GA (1998).** Compares the non-dominated members of the offspring against
the parents to form the next parent set, and proves convergence to the front.

**Original layered ranking + fitness sharing (the immediate predecessor).** Rank the
population into Pareto layers (each layer assigned a dummy fitness kept below the previous
layer's minimum shared value, so earlier layers always dominate selection pressure), then
apply fitness sharing within each layer. The ranking as implemented is `O(M N^3)`, and the
within-layer spread is governed by `sigma_share`.

## Evaluation settings

The yardsticks already in use for comparing MOEAs:

- **Two-objective test problems** chosen from the literature: Schaffer's SCH, Fonseca &
  Fleming's FON, Poloni's POL, Kursawe's KUR, and the ZDT family ZDT1/ZDT2/ZDT3/ZDT4/ZDT6,
  which between them probe convex, non-convex, discontinuous, and many-local-front Pareto
  fronts with up to 30 decision variables.
- **Constrained problems** for the constraint-handling study: CONSTR, SRN, TNK, and a
  five-objective seven-constraint WATER problem.
- **Protocol:** population size 100, run to 25000 function evaluations (≈250 generations),
  summaries computed over ten independent runs. Real-coded variation with SBX
  crossover probability `p_c = 0.9`, `eta_c = 20`; polynomial mutation `p_m = 1/n`,
  `eta_m = 20`. Competing methods use the parameter settings from their own original studies
  (PAES archive 100, depth 4; SPEA 80+20 population/archive).
- **Metrics**, each capturing one of the two algorithm goals:
  - *Convergence* `Upsilon`: pick `H` uniformly spaced points on the true front; for each
    obtained solution take its minimum Euclidean distance to those `H` points; average. Lower
    is better.
  - *Diversity / spread* `Delta`: with consecutive-solution gaps `d_i` (`i = 1..N-1`) along the
    obtained front, mean gap `dbar`, and `d_f`, `d_l` the distances from the front's two
    extreme obtained points to the true extreme solutions,

    ```
    Delta = (d_f + d_l + sum_{i=1}^{N-1} |d_i - dbar|) / (d_f + d_l + (N-1) dbar).
    ```

    A perfectly uniform spread with both extremes covered gives `Delta = 0`; lower is better.

## Code framework

The algorithm plugs into a standard real-coded generational MOEA loop: initialize a population
inside the box bounds, evaluate invalid fitness values, create candidate solutions with bounded
real-coded variation, and return a fixed-size population after each generation. A fitness object
can compare dominance between two individuals; the loop itself leaves the fixed-size survivor
choice open.

```python
from copy import deepcopy
import random

class RealCodedMOEA:
    """A multiobjective evolutionary loop with one generation-strategy slot."""

    def __init__(self, pop_size, n_var, bounds, mate, mutate,
                 cx_prob=0.9, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_var = n_var
        self.bounds = bounds                       # (low, up)
        self.mate = mate
        self.mutate = mutate
        self.cx_prob = cx_prob
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def clone_and_vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        low, up = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.cx_prob:
                self.mate(offspring[i], offspring[i + 1],
                          eta=self.cx_eta, low=low, up=up)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            self.mutate(ind, eta=self.mut_eta, low=low, up=up, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def evaluate_invalid(self, population, evaluate):
        for ind in population:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

    def generation(self, population, evaluate):
        """Return the next fixed-size population."""
        # TODO: choose the next population.
        pass


def run(strategy, population, evaluate, n_gen):
    strategy.evaluate_invalid(population, evaluate)
    for _ in range(n_gen):
        population = strategy.generation(population, evaluate)
    return population
```

The outer loop supplies the population, evaluator, variation primitives, bounds, and dominance
checks; `generation` is the single neutral hook.
