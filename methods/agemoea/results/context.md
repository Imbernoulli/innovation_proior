## Research question

A multi-objective optimization problem asks for decision vectors whose objective vectors
cannot be improved in one objective without worsening another. An evolutionary algorithm does
not return the whole Pareto front; it returns a finite approximation. That approximation
should be close to the true trade-off surface and also distributed across it.

The step under study is the environmental-selection step in an elitist generational algorithm.
After parents and offspring are merged and sorted into non-dominated fronts, whole fronts can
be accepted until one front would overflow the population size. That overflowing front must be
trimmed. The question is how to define the secondary pruning rule that decides which members of
that front survive, given a primary convergence ranking from dominance.

## Background

A real-coded multi-objective evolutionary algorithm usually follows the same loop: initialize
a population, evaluate objective vectors, select parents, apply Simulated Binary Crossover
(SBX) and polynomial mutation, evaluate offspring, merge parents and offspring, then prune the
merged pool back to the target population size. The variation operators are standard; most of
the algorithmic character sits in the survival rule.

Pareto dominance supplies the primary convergence ranking. A solution dominates another if it
is no worse in every objective and strictly better in at least one. Fast non-dominated sorting
partitions the merged pool into fronts `F_1, F_2, ...`: the first front is non-dominated, the
second becomes non-dominated after removing the first, and so on. When the next whole front
does not fit, points inside that same front are mutually non-dominating, so another criterion
must decide which of them survive.

The secondary criterion acts in objective space. On benchmark families such as ZDT, DTLZ, and
WFG, the Pareto front may be convex, disconnected, linear, degenerate, concave, multimodal, or
close to a box-like boundary. Different secondary rules use fixed reference layouts, angle
measures, or axis-aligned crowding to organize points within a front.

Normalization is part of the setting. Objectives can have different offsets and scales, so a
distance computed on raw objective vectors can be dominated by whichever objective has the
largest numerical range. NSGA-III introduced a normalization procedure: estimate the ideal
point by per-objective minima, translate objective vectors by that point, identify extreme
points, fit a hyperplane through them, and use the hyperplane intercepts as per-axis scales. If
the hyperplane is degenerate, a per-axis maximum fallback is used. This is a general way to put
objective vectors on a comparable scale before a secondary rule acts.

## Baselines

**NSGA-II (Deb, Pratap, Agarwal & Meyarivan, 2002).** NSGA-II uses fast non-dominated sorting
for the primary rank and crowding distance as the tie-breaker inside a front. For each
objective, it sorts the front, gives boundary points infinite distance, and adds the
normalized neighbor gap around each interior point; the sum across objectives is maximized. The
rule is cheap and preserves extremes via the axis-aligned crowding estimate.

**MOEA/D (Zhang & Li, 2007).** MOEA/D decomposes the multi-objective problem into scalar
subproblems using weight vectors and scalarizing functions such as Tchebycheff or
penalty-based boundary intersection. It gives a cooperative search structure organized by a
mapping from reference weights to front locations.

**SPEA2 (Zitzler, Laumanns & Thiele, 2001).** SPEA2 combines dominance strength with a
nearest-neighbor density estimate in objective space and keeps an external archive that is
truncated to a fixed size.

**NSGA-III (Deb & Jain, 2014).** NSGA-III was designed for many objectives. It uses structured
reference points, associates solutions with reference directions, and preserves niche counts.
Its normalization by ideal point, extreme points, and hyperplane intercepts puts objectives on
a comparable scale; selection pressure after normalization is organized by perpendicular
distances to the reference directions.

**RVEA and related angle-based methods.** Reference-vector algorithms combine convergence and
angular diversity by measuring each solution relative to reference directions, using the angle
between a solution and its reference direction as the diversity signal.

**Front-modeling methods.** Some methods explicitly fit a parametric model of the current
non-dominated set and then define selection quantities from that model. Nonlinear fitting
routines such as Levenberg-Marquardt are used to fit the model, and the refit can be done once
every several generations rather than continuously.

## Evaluation settings

The natural benchmark families are the standard ones whose true fronts have different shapes:
ZDT for two-objective convex and disconnected fronts, DTLZ for linear and spherical
many-objective fronts, and WFG for convex, degenerate, concave, disconnected, and multimodal
cases. Many-objective settings typically vary the number of objectives, for example
`M = 3, 5, 10` or nearby values, while using the same variation operators across methods.

The usual quality indicators are hypervolume, inverted generational distance, and spread.
Hypervolume rewards dominated volume relative to a reference point, so larger values indicate
better convergence and coverage together. Inverted generational distance measures how far a
sample of the true front is from the approximation set, so lower values are better. Spread
measures uniformity of spacing across the approximation set.

Experimental protocols normally run each algorithm for a fixed budget of generations or
function evaluations over many independent random seeds. Shared operators such as SBX and
polynomial mutation are held fixed so that differences come from environmental selection rather
than variation. Statistical comparisons are then made over the repeated runs.

## Code framework

The open slot sits inside a standard real-coded MOEA harness. The known pieces are
non-dominated sorting, binary tournament selection, SBX crossover, polynomial mutation, and
objective arrays attached to individuals. The undecided piece is only how to prune the first
front that does not fit.

```python
import random
from copy import deepcopy

from deap import tools


class MOEA:
    """Generic generational real-coded multi-objective EA."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        fronts = tools.sortNondominated(population, len(population))
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank
        selected = []
        for _ in range(k):
            a, b = random.sample(population, 2)
            if a._rank < b._rank:
                selected.append(deepcopy(a))
            elif b._rank < a._rank:
                selected.append(deepcopy(b))
            else:
                selected.append(deepcopy(a if random.random() < 0.5 else b))
        return selected

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi,
                indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring
        fronts = tools.sortNondominated(combined, len(combined))
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                # TODO: decide which candidates fill this open slot.
                raise NotImplementedError
        return next_gen
```
