# Context: Population-based search for the whole Pareto front

## Research question

A great many real design problems carry several conflicting objectives at once — minimize cost *and* maximize reliability, minimize weight *and* maximize stiffness. There is no single best solution: improving one objective costs another. The honest answer is a *set* — the Pareto-optimal trade-off surface, the points that cannot be improved in one objective without being worsened in another. A decision-maker wants to see this whole surface, then pick.

The classical move is to collapse the vector of objectives into one scalar — a weighted sum, an ε-constraint, a goal-attainment target — and solve the resulting single-objective problem. That returns one point. To trace the trade-off surface you re-run with many weight settings, and even then a weighted sum can never recover the concave parts of a non-convex front no matter what weights you choose: the supporting-hyperplane geometry only touches the convex hull of the front. So scalarization is both expensive (many runs) and, on non-convex fronts, incomplete.

The question this landscape poses: can a single optimization run return a good approximation of the *entire* Pareto front — many well-converged solutions, spread evenly across the trade-off surface — without asking the user to hand-tune a diversity parameter, and without a runtime cost that explodes with population size?

## Background

**Pareto dominance.** For a minimization problem with objective vector f(x) = (f_1, …, f_M), solution x *dominates* y (written x ≺ y) iff f_i(x) ≤ f_i(y) for every i and f_j(x) < f_j(y) for at least one j. A solution is *nondominated* in a set if nothing in the set dominates it. The Pareto-optimal set is the nondominated set over the whole feasible region; its image in objective space is the Pareto front. Two requirements define a good approximation: **convergence** (the returned points sit on or near the true front) and **diversity** (they are spread uniformly along it, including the extremes).

**Why a population helps.** An evolutionary algorithm carries a whole population of candidate solutions and evolves it by selection, crossover, and mutation. Because the unit of work is already a set, one run can in principle hold many trade-off solutions simultaneously — exactly the object a multi-objective problem demands. The open engineering question is how to assign *fitness* to a vector-valued individual (there is no scalar to maximize) and how to keep the population spread out rather than collapsing onto one part of the front.

**Fitness sharing.** The standard pre-existing device for spreading a population is fitness sharing: each individual's fitness is divided down by the crowd around it. With a sharing function sh(d) = 1 − (d/σ_share)^α for d ≤ σ_share and 0 otherwise, an individual's niche count is Σ_j sh(d_ij) and its shared fitness is (raw fitness)/(niche count). This needs a user-supplied radius σ_share, and it costs a comparison of every pair in a niche, i.e. O(N^2) work.

**Diagnostic facts about nondominated-sorting MOEAs, established before this work.** Nondominated-sorting GAs with sharing had three documented shortcomings, repeatedly noted in the literature of the late 1990s: (1) the nondominated-sorting step ran in O(MN^3) time (M objectives, N population), making it costly at large population sizes; (2) they were non-elitist — a good solution found in one generation could be lost in the next, and elitism had been shown to speed up and stabilize GA convergence; (3) the sharing mechanism needed σ_share, and the quality of the spread depended heavily on its value, which the user had to guess (some dynamic-sizing guidelines existed but no parameter-free method). Elitist successors (external archives, adaptive grids) addressed (2) but reintroduced their own bookkeeping and, in several cases, their own diversity parameters. These are facts about the existing methods, knowable by inspecting them; they motivate the design that follows.

## Baselines

**VEGA — Vector Evaluated GA (Schaffer, 1985).** The first GA for multiple objectives. Split the population into M equal sub-populations; select each sub-population by a single objective; then shuffle them together and apply crossover and mutation. Simple, but each sub-population pulls toward one objective's optimum, so the population speciates toward the individual objective extremes and the middle of the front is starved. It is effectively a weighted-sum selection and inherits the convex-front limitation; it has no concept of dominance.

**MOGA — Multi-Objective GA (Fonseca & Fleming, 1993).** Assign each individual a rank = 1 + (number of population members that dominate it); nondominated members get rank 1. Map rank to fitness, then apply fitness sharing and mating restriction to spread the population. This brought dominance into fitness assignment, but spread still rode on sharing, so it still needed σ_share.

**NSGA — Nondominated Sorting GA (Srinivas & Deb, 1994).** The direct predecessor. Sort the population into successive nondomination fronts: find all nondominated solutions (front 1), assign them a large dummy fitness, then within that front apply fitness sharing to spread them; remove them and repeat on the rest (front 2, with a smaller dummy fitness), and so on. Front-by-front this guarantees solutions on better fronts have higher fitness while sharing diversifies each front. Its three documented limitations are exactly the three diagnostic facts above: the sorting is O(MN^3); it is non-elitist; and the within-front sharing requires σ_share.

**SPEA — Strength Pareto EA (Zitzler & Thiele, 1999).** Elitist. Keep an external population (archive) of all nondominated solutions found so far; it participates in selection. Each archive member's *strength* = how many current-population members it dominates; a current member's fitness = sum of strengths of the archive members that dominate it (so being dominated by many strong solutions is bad). Diversity in the archive is held by deterministic clustering. The naive implementation is O(N^3); with careful bookkeeping O(N^2). It removed the σ_share knob in favor of clustering, but added an archive and a clustering procedure to maintain.

**PAES — Pareto-Archived ES (Knowles & Corne).** A (1+1)-evolution-strategy: one parent, one offspring per step, compared by dominance; ties broken by an adaptive grid over objective space that tracks crowding in 2^d·v subspaces. Archive-based elitism; diversity by grid occupancy. Worst-case O(aMN) with archive length a, overall O(MN^2).

**Rudolph's elitist GA.** Form the combined nondominated set of parent and offspring populations and carry it forward as the next parents; if too small, fill from the offspring. This was proved to converge to the Pareto front, but it carries no explicit diversity mechanism — convergence without spread.

## Evaluation settings

Two-objective test problems drawn from the literature would be the natural yardstick: Schaffer's SCH, Fonseca's FON, Poloni's POL, Kursawe's KUR, and the ZDT family (Zitzler-Deb-Thiele) ZDT1–ZDT6, which were deliberately constructed to stress different difficulties — convex fronts (ZDT1), non-convex fronts (ZDT2), discontinuous fronts (ZDT3), many local fronts (ZDT4), and non-uniform density (ZDT6). For scalable many-objective testing there is the DTLZ family (Deb-Thiele-Laumanns-Zitzler), which lets M grow while the true front stays analytically known. The two quantities a run is scored on are a **convergence** metric (mean distance from the obtained set to the true front) and a **diversity/spread** metric (how evenly the obtained points cover the front, including whether the extremes are reached). The natural points of comparison are the elitist MOEAs PAES and SPEA, run with their authors' recommended settings. Solutions are real-coded with simulated binary crossover and polynomial mutation, or binary-coded with single-point crossover and bitwise mutation.

## Code framework

The primitives that already exist: a `Problem` exposing objective (and constraint) evaluation, real-coded variation operators (simulated binary crossover, polynomial mutation), binary tournament selection, and a generational GA loop. The dominance test and the generic GA skeleton are known. What is *not* yet known — the slots to fill — is how to rank a vector-valued population, how to keep it diverse without a parameter, and how to make the loop elitist.

```python
import numpy as np

def dominates(f_a, f_b):
    # minimization: a dominates b iff a <= b in all and a < b in at least one
    return np.all(f_a <= f_b) and np.any(f_a < f_b)

class Problem:
    n_var: int; n_obj: int; xl: np.ndarray; xu: np.ndarray
    def evaluate(self, X):           # -> F of shape (len(X), n_obj)
        raise NotImplementedError

def sbx_crossover(parents, xl, xu, eta, prob):   # simulated binary crossover (exists)
    ...
def polynomial_mutation(X, xl, xu, eta, prob):   # polynomial mutation (exists)
    ...

def rank_population(F):
    # TODO: split a vector-valued population into ordered nondomination fronts.
    #       Must be cheaper than comparing-and-re-comparing every pair every peel.
    pass

def diversity_metric(F_front):
    # TODO: per-front density estimate that needs NO user parameter,
    #       and is cheaper than every-pair sharing.
    pass

def compare(i, j):
    # TODO: a single ordering that fuses convergence (front) and diversity.
    pass

def survival(pop_F, n_survive):
    # TODO: elitist truncation of a combined parent+offspring set to n_survive,
    #       using rank_population + diversity_metric + compare.
    pass

def run(problem, pop_size, n_gen):
    X = np.random.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    F = problem.evaluate(X)
    for _ in range(n_gen):
        # selection (TODO: by `compare`) -> crossover -> mutation -> offspring
        # TODO: combine parents+offspring, then survival(...) back down to pop_size
        pass
    return X, F
```
