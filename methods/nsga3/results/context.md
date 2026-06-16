## Research question

A problem with several conflicting objectives `f(x) = (f_1(x), ..., f_M(x))` (take all to be
minimized) has no single best solution: improving one objective forces another to get worse,
so the answer is a *set* — the Pareto-optimal set, the feasible points no other feasible
point can beat on every objective at once. `x` dominates `y` when `f_m(x) <= f_m(y)` for all
`m` and `f_m(x) < f_m(y)` for at least one `m`; the Pareto front is the image of the
non-dominated points. The practical aim is to recover, in **one** run, a good approximation
of that whole front — well-converged *onto* it and well-spread *across* it — so a
decision-maker can choose a trade-off afterward.

A population-based evolutionary algorithm is the natural tool because it carries many
candidates at once. By the time this problem is sharp, two- and three-objective problems are
handled well by the dominant elitist EAs. The open difficulty is **many objectives** — four
or more, often ten to fifteen — which arise constantly in real applications with several
stakeholders. The precise question: keep the elitist, fast-sorting framework that already
works at two and three objectives, but make it actually *select* and actually *stay spread*
when `M` is large, where the existing diversity machinery quietly stops doing either.

## Background

The dominant template for an elitist multi-objective EA (MOEA) at this point combines two
ingredients, both reused as given:

- **Fitness by non-domination ranking** (Goldberg 1989). Partition a population into Pareto
  layers: find the non-dominated members (rank 1), set them aside, find the non-dominated
  members of the rest (rank 2), and so on. Selection favors lower ranks. This collapses the
  vector objective into a scalar rank that drives the population toward the front. Fast
  non-dominated sorting (Deb et al. 2002) computes all layers in `O(M N^2)` for population
  `N`.
- **An elitist `(mu+lambda)` survival step.** Merge the `N` parents with the `N` offspring
  into a combined pool `R = P ∪ Q` of size `2N`, sort it into fronts `F_1, F_2, ...`, and fill
  the next population front by front until it overflows. The last front that does not fully
  fit, `F_l`, must be *split*: only `K = N - |∪_{i<l} F_i|` of its members survive, chosen to
  best serve diversity. Carrying the best-so-far parents forward measurably speeds
  convergence and prevents regressions where good solutions are lost between generations.

For real-coded (continuous-variable) problems, two variation operators are standard building
blocks taken as given:

- **Simulated binary crossover, SBX** (Deb & Agrawal 1995). A real-coded recombination that
  imitates single-point binary crossover. With spread factor `beta = |(c2-c1)/(p2-p1)|`,
  children are sampled from the polynomial density `0.5(eta_c+1)beta^{eta_c}` for `beta<=1`
  and `0.5(eta_c+1)/beta^{eta_c+2}` for `beta>1`; inverting the CDF from `u ~ U[0,1]` gives
  `beta_q = (2u)^{1/(eta_c+1)}` for `u<=0.5` else `(1/(2(1-u)))^{1/(eta_c+1)}`, and
  `c1 = 0.5[(1+beta_q)p1 + (1-beta_q)p2]`, `c2 = 0.5[(1-beta_q)p1 + (1+beta_q)p2]`. The
  distribution index `eta_c` controls locality: large `eta_c` keeps children near parents.
- **Polynomial mutation** (Deb & Goyal). For `p in [x_L, x_U]` with normalized bound
  distances, drawing `u ~ U[0,1]` and `mut_pow = 1/(eta_m+1)` gives a bounded step
  `delta_q (x_U - x_L)`; `eta_m` sets the perturbation scale and `p_m = 1/n` mutates about
  one of `n` variables per individual on average.

Several observed facts about how this template behaves *as `M` grows* frame the design
space:

- **Almost everything becomes non-dominated.** It is well documented that in a randomly
  generated population the fraction of non-dominated members grows roughly exponentially with
  the number of objectives. For large `M`, nearly the entire population already lands in the
  first Pareto front. Since a domination-based EA selects by rank, having almost all members
  at rank 1 leaves rank with almost nothing to discriminate on: the selection pressure that
  drives convergence collapses, and the search slows.
- **Diversity estimation becomes expensive and uninformative.** Identifying the neighborhood
  of a solution to gauge crowding gets computationally heavy in high-dimensional objective
  space, and the density estimate itself loses meaning when the points are sparse and almost
  all mutually non-dominated.
- **Recombination loses bite.** When only a handful of solutions populate a large objective
  space, candidate parents tend to be widely separated; recombining two distant parents
  produces offspring that are themselves distant from both, so the recombination operator —
  considered a key search operator in an EA — becomes far less effective.
- **A weighted-sum scalarization does not spread evenly.** It was shown (Das & Dennis 1998)
  that minimizing uniformly spaced weighted combinations of the objectives does *not* yield
  uniformly spaced points on the Pareto front: where the obtained points land depends on the
  curvature of the front, and entire non-convex regions are missed altogether. So a
  many-objective method cannot rely on the simple idea that evenly spaced scalarization
  weights automatically produce evenly spaced trade-off solutions.

## Baselines

Available prior approaches and their gaps:

**NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002) — the framework.** The elitist template
above, with one specific split rule for the overflowing last front `F_l`: a **crowding
distance**. For each member of `F_l`, sort the front by each objective in turn, and add up the
objective-wise normalized gaps to its two neighbors; boundary solutions (the extremes in any
objective) are given infinite crowding distance so they are always kept. Members with the
*largest* crowding distance — the ones in the least crowded regions — are selected to fill
`K`. This is a cheap, approximate density estimate that maintains a good spread at two and
three objectives. **Gap:** the crowding distance is a per-axis neighbor-gap density estimate,
and in many-objective space it degrades — neighbors in one objective are not neighbors in the
full vector, the estimate flattens toward near-uniform when almost everything is
non-dominated, and computing it is costly in high `M`. So the very mechanism that supplies
diversity stops distinguishing solutions exactly where diversity is hardest to maintain.

**MOEA/D (Zhang & Li 2007).** Decompose the multi-objective problem into `N` scalar
subproblems, one per weight vector `w` drawn from a spread set, and solve them cooperatively:
each subproblem is optimized using information from its neighboring subproblems (those with
nearby weight vectors, a neighborhood of size `T`). A subproblem's scalar objective is a
scalarizing function of the weight vector — Tchebycheff `max_i w_i |f_i(x) - z_i^*|` against a
utopian point `z^*`, or a penalty-boundary-intersection (PBI) form
`d_1 + theta·d_2` combining distance along and perpendicular to the reference direction. Each
offspring updates whichever neighboring subproblems it improves. The structured weight set
gives MOEA/D an explicit, predefined notion of where on the front to aim. **Gap:** it leans
on a chosen scalarizing function and the extra knobs that come with it — the neighborhood
size `T`, and for PBI the penalty parameter `theta` (a value of 5 is suggested) — and the
weight-to-Pareto-point mapping again depends on the scalarization and the front's geometry.
It is a different paradigm from a dominance-ranked elitist EA, not a drop-in fix for the
last-front split.

**Earlier diversity-by-sharing schemes (the prior wave).** Rank the population into Pareto
layers, then within each layer spread the population by fitness sharing — degrade an
individual's fitness in proportion to a niche count of neighbors within radius
`sigma_share`. **Gap:** the spread depends sharply on the user-set `sigma_share`, presupposes
a distance metric and an implicit guess at how many niches the front supports, and computing
niche counts compares every pair at `O(N^2)`; there is no parameter-free recipe.

## Evaluation settings

The yardsticks already in use for many-objective MOEAs:

- **Scalable many-objective test problems**: the DTLZ family (DTLZ1-DTLZ4), each definable
  for any number of objectives `M`, with `M - 1 + k` variables (`k = 5` for DTLZ1, `k = 10`
  for the others). They probe a linear front with many local fronts (DTLZ1), a spherical
  concave front (DTLZ2), a stiff multimodal version (DTLZ3), and a biased-density front
  (DTLZ4), at `M` from 3 up to 15.
- **Two- and three-objective continuous problems** of the ZDT type (convex, disconnected
  fronts) and three-objective DTLZ2 used as smaller cases that can still be visualized.
- **Protocol**: real-coded variation with SBX crossover and polynomial mutation; population
  size fixed before the run; runs repeated over many independent seeds; the population
  members of the final generation are used for the measures.
- **Metric**: inverted generational distance, `IGD(A, Z) = (1/|Z|) sum_{z in Z} min_{a in A}
  ||z - a||_2`, the average distance from a dense set `Z` of true-front points to the nearest
  obtained solution `A`. It rolls convergence and diversity into one number; lower is better.

## Code framework

The method plugs into a standard real-coded generational MOEA loop. The loop owns the
population, the bounded real-coded variation operators (SBX crossover, polynomial mutation),
a fitness object that can test dominance between two individuals, and the fast non-dominated
sort. After variation it must return a fixed-size survivor population from the combined pool;
that fixed-size survival choice — specifically how to split the overflowing last front — is
the single open slot.

```python
from copy import deepcopy
import random


class MOEA:
    """A real-coded multi-objective evolutionary loop with one open survival slot.

    Individual interface:
      ind.fitness.values             -> tuple of objective values (minimized)
      ind.fitness.dominates(other.fitness) -> bool
      ind.fitness.valid              -> bool (True once evaluated)
    """

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up)
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        """Choose k parents from the population for mating."""
        # TODO: the mating-selection rule (if any).
        pass

    def vary(self, parents):
        """SBX crossover + polynomial mutation on clones (invalidate fitness)."""
        offspring = [deepcopy(ind) for ind in parents]
        low, up = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            cx_simulated_binary_bounded(offspring[i], offspring[i + 1],
                                        eta=self.cx_eta, low=low, up=up)
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            mut_polynomial_bounded(ind, eta=self.mut_eta, low=low, up=up,
                                   indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        """Environmental selection from the combined 2N pool back to pop_size.
        Fronts that fully fit are kept; the overflowing last front is split."""
        # TODO: the last-front split rule we will design.
        pass


def run(strategy, population, evaluate, n_gen):
    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.values = evaluate(ind)
    for _ in range(n_gen):
        parents = strategy.select(population, strategy.pop_size)
        offspring = strategy.vary(parents)
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        population = strategy.survive(population, offspring)
    return population
```

The outer loop supplies the population, evaluator, variation primitives, bounds, and
dominance checks; `survive` (the last-front split) is the open hook.
