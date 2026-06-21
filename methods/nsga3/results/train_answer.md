A problem with several conflicting objectives $\min f(x) = (f_1(x), \dots, f_M(x))$ has no single best solution: improving one objective forces another to worsen, so the answer is the Pareto-optimal set, and the practical aim is to recover in one run a good approximation of the whole front — converged onto it and spread across it — so a decision-maker can pick a trade-off afterward. A population-based elitist evolutionary algorithm is the natural tool, and at two or three objectives the dominant template works beautifully: merge the $N$ parents with the $N$ offspring into a pool $R$ of size $2N$, fast-non-dominated-sort it into Pareto layers $F_1, F_2, \dots$, fill the next population front by front until it overflows, and split the last front $F_l$ that does not fully fit by crowding distance, keeping the $K$ members in the least crowded regions. Convergence comes from the rank, diversity from the crowding distance. The trouble is many objectives — four or more, often ten to fifteen — where this quietly stops working, and it is worth being precise about why before touching anything.

Two things break as $M$ grows. First, selection pressure collapses: the fraction of mutually non-dominated members in a random population climbs roughly exponentially with $M$, so by $M=10$ almost the entire pool already sits in front 1. Rank, the engine that pushes the population onto the true front, has nothing left to discriminate on — nearly everyone is rank 1 — and the convergence engine idles. Domination simply cannot separate points in high dimensions; nothing on the diversity side fixes that. But it carries a consequence: when $F_l$ is essentially the whole pool, the entire burden of *who survives* shifts onto the last-front split, so that split had better be right. And the second failure is exactly there. Crowding distance, for each member of $F_l$, sorts the front by each objective in turn and sums the normalized gaps to its two immediate neighbors along each axis. In 2D the per-axis neighbor really is your neighbor on the front, so this is a genuine cheap density estimate. With ten axes it is not: "nearest neighbor along axis 3" has almost nothing to do with "nearest neighbor in the full ten-dimensional vector," and when $N$ points are scattered sparsely in a ten-dimensional box the axiswise gaps become large and roughly equal for everyone, so the crowding distances flatten toward near-uniform and the split degenerates toward random — while also becoming expensive, since neighborhood estimation in high dimensions is costly. The wall is therefore this: I cannot maintain spread by *estimating* the population's own density when the population lives in a space where density estimates are meaningless. The available alternatives do not escape it cleanly either — MOEA/D leans on a chosen scalarizing function and its extra knobs (neighborhood size $T$, the PBI penalty $\theta$) and is a different paradigm rather than a drop-in fix, and earlier fitness-sharing schemes hinge on a user-set niche radius $\sigma_{\text{share}}$ and an $O(N^2)$ pairwise count with no parameter-free recipe.

I propose NSGA-III, which keeps NSGA-II's entire elitist framework — combined pool, fast sort, fill fronts until overflow — and replaces only the last-front split, swapping crowding distance for reference-point niching. The governing idea is to stop *estimating* diversity from a sparse cloud and instead *impose* it from a fixed external set of targets decided ahead of time and independently of the population, so it can never flatten or degrade no matter how high $M$ climbs. The targets are a grid of evenly spaced directions. Where to put them is a question the scalarization world already half-answered: Das and Dennis showed that sweeping uniformly spaced *weighted sums* of the objectives does not produce uniformly spaced front points — the spacing depends on the front's curvature and every non-convex region is missed outright, because no weighted sum is ever minimized in a dent — so that route is poisoned. But the same work gives a scale-independent way to lay evenly spaced points on the unit simplex $\sum_i w_i = 1$, $w_i \ge 0$: with $p$ divisions per axis, enumerate every vector whose components lie in $\{0/p, 1/p, \dots, p/p\}$ and sum to one. By stars-and-bars there are $H = \binom{M+p-1}{p}$ of them, perfectly uniform by construction, living on the abstract simplex and so independent of the front's shape. I use these not as weights to scalarize but as a fixed set of reference points; the ray from the origin through each is a reference *line*, a direction I want covered. One subtlety forces a refinement at large $M$: writing a grid point as integer counts $k_i/p$ with $\sum_i k_i = p$, an interior point (every $k_i \ge 1$) exists only when $p \ge M$, and the interior fraction $\binom{p-1}{M-1}/\binom{M+p-1}{M-1}$ collapses fast, so a single small-$p$ grid puts almost all targets on boundary faces. The fix keeps the same generator and, when interior directions are wanted, shrinks an additional layer toward the simplex center (scale the layer and re-center) before taking the union.

To compare population members against directions defined on the clean $[0,1]$ simplex, I must normalize the population into that frame every generation, because the objectives run on wildly different and drifting scales as the front converges. I build the frame from the current survivors $S_t$. The ideal point is the per-objective best, $z_i^{\min} = \min_{S_t} f_i$, and I translate so it becomes the origin, $f_i'(s) = f_i(s) - z_i^{\min}$. The top of the frame is the linear hyperplane through the $M$ per-axis extreme points, whose axis intercepts give the per-axis scale. Choosing each extreme robustly matters: plain $\arg\max f_j'$ can pick a dominated outlier that is large in $j$ only because it is large everywhere. Instead I want the point that is extreme *along the $j$-axis specifically* — large in $j$, small elsewhere — which the achievement scalarizing function delivers,
$$\mathrm{ASF}(s, w^j) = \max_i \frac{f_i'(s)}{w_i^j}, \qquad w_j^j = 1,\ w_i^j = \epsilon\ (i \ne j),$$
so the tiny off-axis weights make any off-axis coordinate expensive — $f_i'/\epsilon$ dominates the max unless $f_i'$ is nearly zero — and $z^{j,\max} = \arg\min_{S_t} \mathrm{ASF}(s, w^j)$ lands closest to the $j$ axis. In code the division by $\epsilon$ is a multiplication of off-axis coordinates by $1/\epsilon = 10^6$, which preserves the ordering. Stacking the translated extremes as rows of $A$ and writing the hyperplane as $A x = \mathbf{1}$, the coefficient $x_j$ is the reciprocal axis-intercept length, so the intercept scale is $a_j = 1/x_j$ from one matrix solve, with guards: a singular $A$ falls back to the current per-axis worst, and a zero in $x$, a failed reconstruction $A x = \mathbf{1}$, a nonpositive intercept, or an intercept overshooting the worst point all fall back to the per-axis worst of the fronts being normalized. The normalization is then
$$f_i^n(s) = \frac{f_i(s) - z_i^{\min}}{a_i},$$
which maps the ideal to $0$ and each axis intercept to $1$, dropping the front roughly onto the same simplex the reference grid lives on; recomputing it every generation re-fits the grid to wherever the population currently is, which is what handles differently-scaled, drifting fronts with no tuning.

With everyone normalized I associate each solution to a reference direction, and the crux of the whole design is that I associate to the *line*, not the nearest reference *point*. A reference point names a direction; a solution close to the origin and one far out along the same ray serve that direction equally well for diversity, and their difference — position along the ray — is convergence, which is the rank's job. So the right measure is the perpendicular distance from the solution to the reference line,
$$d_\perp(s, w) = \left\| s^n - \frac{s^n \cdot w}{\|w\|^2}\, w \right\|,$$
and I record both the minimizing reference point $\pi(s)$ and that distance $d(s)$. This cleanly factors the two goals: along-ray position is convergence (front rank), perpendicular offset is which-direction-am-I-serving (the split) — the decoupling crowding distance never had. The split itself fills $K$ vacancies from $F_l$ while fronts $1,\dots,l-1$, already in $P_{t+1}$, are associated to directions. Let $\rho_j$ be the niche count, the number of already-selected members associated to reference point $j$. To get a uniform spread I pour survivors into starving directions first: repeatedly find the reference point with the smallest $\rho_j$ (ties broken at random) and give it an $F_l$ member. The two cases differ on purpose — if $\rho_j = 0$ the added member becomes the direction's sole ambassador, so I take the associated $F_l$ member with the *shortest* $d_\perp$, the one lying most squarely on the ray; if $\rho_j \ge 1$ the direction already has a representative and I just need another body, so I take a *random* associated member, spending the effort of the closest pick only when establishing a brand-new niche. Increment $\rho_j$, remove the member, drop any reference point with no remaining $F_l$ member for this generation, and repeat $K$ times.

Because diversity is now entirely a survival property, the edges of the algorithm simplify. Mating selection in NSGA-II would use a binary tournament on rank-then-crowding — but crowding distance is the very thing I discarded, and the spread it would protect is already protected by the survival niching, so I draw parents at random from $P_{t+1}$ and let variation explore, with no fragile mating-diversity knob. Variation is the standard SBX-plus-polynomial-mutation pair, but the high-dimensional recombination failure mode — distant parents producing offspring distant from both, landing in unexplored voids — demands a response: I push the SBX distribution index up to $\eta_c = 30$, larger than the usual 20, so children stay near their parents, acting like a soft mating restriction that recombines locally; crossover probability is $1$, polynomial mutation keeps $\eta_m = 20$ and $p_m = 1/n$. The only structural choices are the reference grid (resolution $H$, a user knob, not an algorithmic one) and the population size $N \sim H$. The cost stays in NSGA-II's ballpark: the sort is $O(N \log^{M-2} N)$, normalization $O(M^2 N)$ for extremes plus $O(M^3)$ for one solve, association $O(M N H)$, niching $O(N H)$, giving $O(N^2 \log^{M-2} N)$ or $O(N^2 M)$ — with no high-dimensional density estimate to break the split.

```python
import numpy
from collections import namedtuple
from itertools import chain


NSGA3Memory = namedtuple("NSGA3Memory", ["best_point", "worst_point", "extreme_points"])


def uniform_reference_points(nobj, p=4, scaling=None):
    """Das-Dennis points on sum w_i = 1; optional scaling shrinks a layer
    toward the simplex center so several layers can be combined."""
    def recurse(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(recurse(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = numpy.array(recurse(numpy.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj
    return ref_points


def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """Per axis j, minimize ASF(s,w^j)=max_i (s_i-best_i)/w_i^j.
    DEAP implements w_i^j=1e-6 off axis as a 1e6 multiplier."""
    if extreme_points is not None:
        fitnesses = numpy.concatenate((fitnesses, extreme_points), axis=0)
    ft = fitnesses - best_point
    asf = numpy.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = numpy.max(ft * asf[:, numpy.newaxis, :], axis=2)
    return fitnesses[numpy.argmin(asf, axis=1), :]


def find_intercepts(extremes, ideal, current_worst, front_worst):
    """Axis-intercept scales of the hyperplane through the M extreme points:
    solve A x = 1 with A = extremes - ideal, then a_j = 1/x_j."""
    b = numpy.ones(extremes.shape[1])
    A = extremes - ideal
    try:
        x = numpy.linalg.solve(A, b)
    except numpy.linalg.LinAlgError:
        return current_worst
    if numpy.count_nonzero(x) != len(x):
        return front_worst
    intercepts = 1 / x
    if (not numpy.allclose(numpy.dot(A, x), b)
            or numpy.any(intercepts <= 1e-6)
            or numpy.any((intercepts + ideal) > current_worst)):
        return front_worst
    return intercepts


def associate_to_niche(F, ref_points, ideal, intercepts):
    """Normalize with DEAP's intercept convention, then assign each member
    to the reference line of minimum perpendicular distance."""
    fn = (F - ideal) / (intercepts - ideal + numpy.finfo(float).eps)
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(ref_points), axis=1)
    norm = numpy.linalg.norm(ref_points, axis=1)
    d = numpy.sum(fn * ref_points, axis=2) / norm.reshape(1, -1)
    proj = d[:, :, numpy.newaxis] * ref_points[numpy.newaxis, :, :] / norm[numpy.newaxis, :, numpy.newaxis]
    dist = numpy.linalg.norm(proj - fn, axis=2)        # perpendicular distance to each line
    niches = numpy.argmin(dist, axis=1)
    dist = dist[range(niches.shape[0]), niches]
    return niches, dist


def niching(last_front, k, niches, distances, niche_counts):
    """Fill k slots from the last front, serving the smallest-niche-count
    reference point first (closest member if empty, else random)."""
    selected = []
    available = numpy.ones(len(last_front), dtype=bool)
    while len(selected) < k:
        live = numpy.zeros(len(niche_counts), dtype=bool)
        live[numpy.unique(niches[available])] = True
        min_count = numpy.min(niche_counts[live])
        chosen_refs = numpy.flatnonzero(live & (niche_counts == min_count))
        numpy.random.shuffle(chosen_refs)
        for j in chosen_refs[: k - len(selected)]:
            members = numpy.flatnonzero((niches == j) & available)
            if niche_counts[j] == 0:
                sel = members[numpy.argmin(distances[members])]   # new niche -> closest
            else:
                numpy.random.shuffle(members)
                sel = members[0]                                  # filled niche -> random
            available[sel] = False
            niche_counts[j] += 1
            selected.append(last_front[sel])
    return selected


def selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    """DEAP-style environmental selection: keep whole fronts, split the last
    by reference-point niching."""
    if nd == "standard":
        pareto_fronts = sortNondominated(individuals, k)
    elif nd == "log":
        pareto_fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selNSGA3: invalid non-dominated sorting choice")

    fitnesses = numpy.array([ind.fitness.wvalues for f in pareto_fronts for ind in f])
    fitnesses *= -1.0                       # always handle selection as minimization

    if best_point is not None and worst_point is not None:
        best_point = numpy.min(numpy.concatenate((fitnesses, best_point), axis=0), axis=0)
        worst_point = numpy.max(numpy.concatenate((fitnesses, worst_point), axis=0), axis=0)
    else:
        best_point = numpy.min(fitnesses, axis=0)
        worst_point = numpy.max(fitnesses, axis=0)

    extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
    front_worst = numpy.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
    intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
    niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    idx, cnt = numpy.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
    niche_counts[idx] = cnt

    chosen = list(chain(*pareto_fronts[:-1]))
    n = k - len(chosen)
    chosen.extend(niching(pareto_fronts[-1], n,
                          niches[len(chosen):], dist[len(chosen):], niche_counts))

    if return_memory:
        return chosen, NSGA3Memory(best_point, worst_point, extreme_points)
    return chosen
```

Deployment shell (NSGA-II framework, reference-point survival):

```python
import random
from copy import deepcopy


class NSGA3:
    """NSGA-III: NSGA-II framework with reference-point niching survival."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None,
                 ref_points=None, ref_divisions=4, ref_scaling=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.bounds = bounds
        self.cx_eta = cx_eta                       # large SBX index -> children near parents
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.ref_points = (ref_points if ref_points is not None
                           else uniform_reference_points(n_obj, ref_divisions, ref_scaling))

    def select(self, population, k):
        sel = [deepcopy(ind) for ind in population]   # no mating selection (diversity in survive)
        random.shuffle(sel)
        return sel[:k]

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            cx_simulated_binary_bounded(offspring[i], offspring[i + 1],
                                        eta=self.cx_eta, low=lo, up=hi)
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            mut_polynomial_bounded(ind, eta=self.mut_eta, low=lo, up=hi,
                                   indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring             # elitist (mu + lambda)
        return selNSGA3(combined, self.pop_size, self.ref_points)
```
