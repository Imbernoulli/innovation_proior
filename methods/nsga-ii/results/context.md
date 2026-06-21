# Context: Population-based search for the whole Pareto front

## Research question

A great many real design problems carry several conflicting objectives at once — minimize cost *and* maximize reliability, minimize weight *and* maximize stiffness. There is no single best solution: improving one objective costs another. The honest answer is a *set* — the Pareto-optimal trade-off surface, the points that cannot be improved in one objective without being worsened in another. A decision-maker wants to see this whole surface, then pick.

The classical move is to collapse the vector of objectives into one scalar — a weighted sum, an ε-constraint, a goal-attainment target — and solve the resulting single-objective problem. That returns one point. To trace the trade-off surface, the run is repeated with many weight settings.

The question this landscape poses: can a single optimization run return a good approximation of the *entire* Pareto front — many well-converged solutions spread across the trade-off surface?

## Background

**Pareto dominance.** For a minimization problem with objective vector f(x) = (f_1, …, f_M), solution x *dominates* y (written x ≺ y) iff f_i(x) ≤ f_i(y) for every i and f_j(x) < f_j(y) for at least one j. A solution is *nondominated* in a set if nothing in the set dominates it. The Pareto-optimal set is the nondominated set over the whole feasible region; its image in objective space is the Pareto front. Two requirements define a good approximation: **convergence** (the returned points sit on or near the true front) and **diversity** (they are spread uniformly along it, including the extremes).

**Why a population helps.** An evolutionary algorithm carries a whole population of candidate solutions and evolves it by selection, crossover, and mutation. Because the unit of work is already a set, one run can in principle hold many trade-off solutions simultaneously — exactly the object a multi-objective problem demands. The open engineering question is how to assign *fitness* to a vector-valued individual (there is no scalar to maximize) and how to keep the population spread out rather than collapsing onto one part of the front.

**Fitness sharing.** The standard pre-existing device for spreading a population is fitness sharing: each individual's fitness is divided down by the crowd around it. With a sharing function sh(d) = 1 − (d/σ_share)^α for d ≤ σ_share and 0 otherwise, an individual's niche count is Σ_j sh(d_ij) and its shared fitness is (raw fitness)/(niche count). This requires a user-supplied radius σ_share, and it costs a comparison of every pair in a niche, i.e. O(N^2) work.

## Baselines

**VEGA — Vector Evaluated GA (Schaffer, 1985).** The first GA for multiple objectives. Split the population into M equal sub-populations; select each sub-population by a single objective; then shuffle them together and apply crossover and mutation. It has no concept of dominance.

**MOGA — Multi-Objective GA (Fonseca & Fleming, 1993).** Assign each individual a rank = 1 + (number of population members that dominate it); nondominated members get rank 1. Map rank to fitness, then apply fitness sharing and mating restriction to spread the population. This brought dominance into fitness assignment and uses σ_share for spreading.

**NSGA — Nondominated Sorting GA (Srinivas & Deb, 1994).** The direct predecessor. Sort the population into successive nondomination fronts: find all nondominated solutions (front 1), assign them a large dummy fitness, then within that front apply fitness sharing to spread them; remove them and repeat on the rest (front 2, with a smaller dummy fitness), and so on. Front-by-front this guarantees solutions on better fronts have higher fitness while sharing diversifies each front. It uses O(MN^3) sorting, is non-elitist, and the within-front sharing requires σ_share.

**SPEA — Strength Pareto EA (Zitzler & Thiele, 1999).** Elitist. Keep an external population (archive) of all nondominated solutions found so far; it participates in selection. Each archive member's *strength* = how many current-population members it dominates; a current member's fitness = sum of strengths of the archive members that dominate it. Diversity in the archive is held by deterministic clustering. The naive implementation is O(N^3); with careful bookkeeping O(N^2).

**PAES — Pareto-Archived ES (Knowles & Corne).** A (1+1)-evolution-strategy: one parent, one offspring per step, compared by dominance; ties are broken by an adaptive objective-space grid whose depth controls the crowding resolution. Archive-based elitism; diversity by grid occupancy. With archive length a, the worst case is O(aMN), which becomes O(MN^2) when the archive is proportional to the population size.

**Rudolph's elitist GA.** Form the combined nondominated set of parent and offspring populations and carry it forward as the next parents; if too small, fill from the offspring. This was proved to converge to the Pareto front, with no explicit diversity mechanism.

## Evaluation settings

Two-objective test problems drawn from the literature would be the natural yardstick: Schaffer's SCH, Fonseca's FON, Poloni's POL, Kursawe's KUR, and the ZDT family instances ZDT1, ZDT2, ZDT3, ZDT4, and ZDT6, which stress convex fronts, non-convex fronts, discontinuous fronts, many local fronts, and non-uniform density. Constrained settings would include benchmark problems such as SRN and TNK, plus larger engineering-style constrained cases such as WATER. The two quantities a run is scored on are a **convergence** metric (mean distance from the obtained set to the true front) and a **diversity/spread** metric (how evenly the obtained points cover the front, including whether the extremes are reached). The natural points of comparison are the elitist MOEAs PAES and SPEA, run with their recommended settings. Solutions are real-coded with simulated binary crossover and polynomial mutation, or binary-coded with single-point crossover and bitwise mutation.

## Code framework

The available primitives are a `Problem` exposing objective evaluation and optional aggregate constraint violation, real-coded variation operators (simulated binary crossover, polynomial mutation), binary tournament selection, and a generational GA loop. The empty slots are where the multi-objective fitness machinery and the survivor selection go.

```python
import numpy as np

def dominates(f_a, f_b):
    # minimization: a dominates b iff a <= b in all and a < b in at least one
    return np.all(f_a <= f_b) and np.any(f_a < f_b)

class Problem:
    n_var: int; n_obj: int; xl: np.ndarray; xu: np.ndarray
    def evaluate(self, X):           # -> F, or (F, CV) with one CV per solution
        raise NotImplementedError

def sbx_crossover(parents, xl, xu, eta, prob):   # simulated binary crossover (exists)
    ...
def polynomial_mutation(X, xl, xu, eta, prob):   # polynomial mutation (exists)
    ...

def rank_population(F, CV=None):
    # TODO: split a vector-valued population into ordered nondomination fronts.
    pass

def diversity_metric(F_front):
    # TODO: per-front density estimate.
    pass

def compare(i, j, ranks, diversity, F=None, CV=None):
    # TODO: an ordering on individuals used by selection and survival.
    pass

def survival(candidate_F, n_survive, candidate_CV=None):
    # TODO: choose n_survive objective vectors to carry to the next generation.
    pass

def run(problem, pop_size, n_gen):
    X = np.random.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    F = problem.evaluate(X)
    for _ in range(n_gen):
        # TODO: selection by `compare` -> crossover -> mutation -> offspring
        # TODO: form the candidate pool for the next generation and call survival(...)
        pass
    return X, F
```
