## Research Question

A multi-objective minimization problem asks for decisions whose objective vectors
`f(x) = (f_1(x), ..., f_m(x))` cannot usually be ordered by one scalar loss. The useful
output is a finite approximation to the Pareto front: it should move close to the true
front and cover its trade-off surface without losing extremes.

Elitist EMOAs handle survival in two parts: respecting Pareto dominance and maintaining
diversity among solutions that dominance cannot separate. The common algorithms address
the second part with density surrogates: crowding distance, nearest-neighbour density,
clustering, or grid occupancy.

The open design question is how a simple elitist EMOA can use a set quality indicator
to guide within-front survival decisions.

## Background

For minimization, a vector `a` weakly dominates `b` when `a_i <= b_i` for every objective,
and it dominates `b` when at least one inequality is strict. A Pareto front approximation
is therefore naturally partitioned into nondominated-sorting levels: the first front is
nondominated in the whole pool, the second is nondominated after removing the first, and
so on.

One unary quality indicator was already central: the dominated hypervolume, also called
the S-metric. Given a reference point `r` that is worse than the objective vectors of
interest, the hypervolume of a set `A` is

```text
S(A, r) = lambda({ y : exists a in A with a <= y <= r }).
```

This indicator rewards both convergence and coverage: moving a point toward the Pareto
front expands dominated volume, and spreading points over different trade-offs prevents
the same dominated region from being counted redundantly. It also depends on a reference
point, and that dependence is concentrated at boundary points whose dominated boxes reach
toward `r`.

Exact hypervolume routines are efficient in two objectives and still manageable in three,
but the cost grows sharply with the objective dimension.

## Baselines

NSGA-II uses fast nondominated sorting and fills the next generation front by front. If
the last accepted front does not fit, it keeps points with high crowding distance: for
each objective, neighbouring gaps are normalized and summed, while boundary points are
protected. The method is parameter-light and robust.

SPEA2 assigns each individual a strength based on how many other individuals it dominates,
adds a k-th-nearest-neighbour density term, and uses an archive truncation procedure to
preserve spread when the archive is too large. It improves on the first strength-Pareto
archive, especially when many individuals are nondominated.

IBEA makes the indicator idea explicit through binary indicators. It assigns fitness from
pairwise indicator values, often an epsilon indicator or a hypervolume-difference indicator,
with an exponential scaling constant. This brings indicators into selection through a scalar
fitness derived from pairwise comparisons.

Adaptive archiving methods show that retaining a bounded nondominated archive can be tied
to dominated-volume arguments.

## Evaluation Settings

The standard benchmark setting uses ZDT problems for two objectives and DTLZ problems for
three objectives, with real-coded SBX recombination and polynomial mutation inherited from
the Deb benchmark setup. Population sizes around 100 and fixed function-evaluation budgets
are typical. Reported quality is dominated hypervolume plus a convergence-distance measure;
successful runs are those that cover the true front rather than collapsing to one part of it.

For two objectives, the reference point matters mainly for the two extremal points of a
nondominated front. For more objectives, many boundary points can contain a worst coordinate,
so a reference point has to be handled during the run rather than treated as only an
after-the-fact reporting choice.

## Code Framework

The surrounding optimizer can be an ordinary steady-state elitist EMOA. It keeps a fixed
population, creates one new evaluated child, appends it to the population, and invokes a
single reduction hook that removes one member from the `mu + 1` pool.

```python
class SteadyStateMOEA:
    def __init__(self, pop_size, make_child):
        self.pop_size = pop_size
        self.make_child = make_child

    def step(self, population):
        child = self.make_child(population)
        pool = population + [child]
        return self.reduce(pool)

    def reduce(self, pool):
        # TODO: choose one member to discard from the fixed-size survivor pool.
        raise NotImplementedError
```

The hook receives evaluated individuals and can read each objective vector. Everything else
is deliberately ordinary: initialization, mating, variation, and objective evaluation are
outside the hook.
