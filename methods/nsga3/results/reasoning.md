Let me start from what actually breaks. I have an elitist multi-objective EA that works beautifully at two or three objectives: merge the N parents with the N offspring into a pool R of size 2N, sort it into Pareto fronts F_1, F_2, ..., and fill the next population front by front until it overflows; the last front that doesn't fully fit, F_l, I split by crowding distance, keeping the K members in the least crowded regions. Convergence comes from the rank, diversity from the crowding distance. Now someone hands me a problem with ten objectives, and the whole thing quietly stops working, and I want to be precise about why before I touch anything.

The first thing that goes is selection pressure. With more objectives, a randomly drawn population has an ever larger fraction of mutually non-dominated members — it climbs roughly exponentially in M. By the time M is ten, almost the entire combined pool is already in front 1. So fast non-dominated sorting, which I rely on to discriminate, returns one giant front and a few stragglers. Rank, which is the thing that pushes the population down onto the true front, has almost nothing left to say: nearly everyone is rank 1. The convergence engine idles. There's nothing I can do about *that* from the diversity side — domination just doesn't separate points in high dimensions — but it tells me something important: when M is large, the burden of actually choosing who survives shifts almost entirely onto the last-front split. If F_l is essentially the whole pool, then "how I split F_l" *is* the algorithm. So I'd better get that split right, and crowding distance is exactly where I think it fails.

Why does crowding distance fail here? Recall what it is: for each member of F_l, I sort the front by each objective in turn and sum the normalized gaps to its two immediate neighbors along each axis, with the boundary points in each objective getting infinite distance so they're always kept. In 2D this is a genuine, cheap density estimate — neighbors along f_1 really are your neighbors on the front. But picture ten axes. "Nearest neighbor along axis 3" has almost nothing to do with "nearest neighbor in the full ten-dimensional vector"; the per-axis neighbor of a point can be arbitrarily far from it in every other objective. So the sum-of-axiswise-gaps stops measuring real crowding. Worse, when points are sparse in a high-dimensional box — which they are, N points scattered in ten dimensions — the gaps along each axis become large and roughly comparable for everyone, so the crowding distances flatten toward near-uniform and the split degenerates into something close to random. And it isn't even cheap anymore: estimating each point's neighborhood in high dimensions is expensive. So the diversity mechanism is both uninformative and costly precisely in the regime where the split has become the whole game. That's the wall: I cannot maintain spread by *estimating* the population's own density when the population lives in a space where density estimates are meaningless.

So let me ask the question differently. The reason density estimation is hard is that I'm trying to *infer* where the gaps in the front are from the few points I happen to have. What if I don't infer it at all? What if I decide, ahead of time and independently of the population, the set of places on the front I want covered, and then just push the population to cover them? Diversity would no longer be something I measure off a sparse cloud; it would be something I *impose* from outside, fixed and known, so it can't flatten or degrade no matter how high M climbs. The question becomes: where exactly do I want coverage, and how do I make the survival step honor it?

"Where on the front do I want coverage" is a question I've seen answered before, in the scalarization world. If I scalarize — collapse the vector objective into min over x of a weighted combination — and sweep the weights, I get different Pareto points. The naive hope is: spread the weights uniformly, get a uniformly spread front. But Das and Dennis showed that's just false. Minimizing uniformly spaced weighted sums of the objectives gives points whose spacing on the front depends entirely on the curvature of the front, bunching up where it's flat and gapping where it's curved, and — fatally — missing every non-convex region of the front outright, because no weighted sum is ever minimized in a non-convex dent. So I can't get even coverage by spacing weights and minimizing weighted sums. That route is poisoned.

But Das and Dennis gave me something else in the same breath that I *can* use, and it's the geometric half, not the scalarization half. They constructed a systematic, scale-independent way to lay down evenly spaced points on the unit simplex — the hyperplane sum_i w_i = 1 with w_i >= 0. Take p divisions along each axis and enumerate every weight vector whose components are in {0/p, 1/p, ..., p/p} and sum to one. By a stars-and-bars count there are H = C(M+p-1, p) of them, and they tile the simplex with equal nearest-neighbor spacing. That's a *grid of directions*, dead simple to generate, perfectly uniform by construction, and it doesn't depend on the front's shape at all because it lives on the abstract simplex, not on the front. This is the "where I want coverage" I was after — not as weights to scalarize, but as a fixed set of target directions in objective space. Let me write the generator, because it's just a recursion: at each objective, hand out i/p of the remaining budget and recurse on the rest, until the last objective takes whatever's left.

  def uniform_reference_points(M, p):
      def recurse(ref, left, depth):
          if depth == M - 1:
              ref[depth] = left / p
              return [ref]
          out = []
          for i in range(left + 1):
              ref[depth] = i / p
              out += recurse(ref.copy(), left - i, depth + 1)
          return out
      return array(recurse(zeros(M), p, 0))

These H points sit on the hyperplane that cuts each objective axis at 1. I'll call each one a reference point, and the ray from the origin through it a reference *line* (a direction). The plan, then: keep NSGA-II's entire framework — combined pool, fast sort, fill fronts until overflow — and replace only the F_l split. The new split should hand survivors to reference directions so that every direction ends up with a representative, which would mean the survivors are spread out exactly as uniformly as the reference grid.

Before I design the split, let me stress-test the reference grid itself, because something nags at me about large M. With p divisions, how many of those H points actually land in the *interior* of the simplex versus on its boundary faces, where one or more w_i are zero? Write a grid point as integer counts k_i/p with sum_i k_i = p. Interior means every k_i >= 1. If p < M there is no such point. If p >= M, subtract one from every component; the remaining p - M units can be distributed among M components in C((p - M) + M - 1, M - 1) = C(p - 1, M - 1) ways. The total grid has C(M + p - 1, M - 1) = C(M + p - 1, p) points, so the interior fraction is C(p - 1, M - 1) / C(M + p - 1, M - 1), and it collapses fast as M grows for fixed modest p. To get even a single interior point I need p >= M. So if I want a manageable number of directions in many objectives, a single small-p grid puts almost all targets on boundary faces. I can keep the same combinatorial generator and, when I need interior directions too, shrink an additional layer toward the simplex center before taking the union. The core selection rule does not change; the reference set is still just a fixed set of directions. Back to the split.

Here's the immediate problem with "hand survivors to reference directions": the reference grid lives on a clean simplex where objectives run 0 to 1, but my actual objectives are on wildly different and shifting scales — one might run in the thousands, another in fractions, and the front itself drifts and rescales as the population evolves. If I measure a solution's closeness to a reference direction in raw objective units, the large-magnitude objective dominates and the directions become meaningless. So I have to normalize the population into the same frame as the reference grid, and I have to do it *every generation*, because the scales move. Let me build that normalization from what's actually available — the current survivors S_t (all the fronts I'm keeping, including F_l).

The bottom corner of the frame is the ideal point: for each objective, the best (minimum) value anyone in S_t achieves, z_i^min = min over S_t of f_i. Translate so the ideal becomes the origin: f_i'(s) = f_i(s) - z_i^min. Now I need the top of the frame — where each axis should be scaled to 1. The natural choice is the linear hyperplane through the M extreme points of the current front, one extreme per objective, and the axis intercepts of that hyperplane give me the per-axis scale. So: how do I pick "the extreme solution along objective j" robustly? My first instinct is just argmax of f_j', the solution with the largest value in objective j. But that's fragile — it can pick a solution that's huge in j only because it's also huge everywhere, a dominated outlier off in a corner, not a genuine spread-defining extreme. I want the solution that is extreme *along the j-axis specifically*, large in j and small in everything else. The achievement scalarizing function does exactly that: define ASF(s, w) = max_i f_i'(s)/w_i, and take the weight vector w^j with w_j^j = 1 and w_i^j = epsilon for i != j. The tiny off-axis weights make any off-axis coordinate expensive: f_i'/epsilon dominates the max unless f_i' is almost zero, while the j coordinate is divided by one. Minimizing this ASF finds the point closest to the j axis. In code I do not actually divide by epsilon; I multiply the off-axis translated coordinates by 1/epsilon = 1e6 and leave the j coordinate alone, which is the same ordering. So z^{j,max} = argmin over S_t of ASF(s, w^j).

Now the M extreme points define a hyperplane, and I want its axis intercepts. I stack the translated extreme points as rows of a matrix A and write the hyperplane in translated coordinates as A x = 1, where 1 is the all-ones vector. The coefficient x_j is the reciprocal of the j-th axis-intercept length, so the translated intercept scale is a_j = 1/x_j. One matrix solve. I have to guard it exactly where the solve can become untrustworthy: if A is singular I use the current per-axis worst values as the scale; if x has a zero, the solved intercepts are nonpositive, the reconstructed plane fails A x = 1, or any intercept plus the ideal overshoots the current worst point, I fall back to the per-axis worst values of the fronts being normalized. With the intercept scales in hand the normalization is

  f_i^n(s) = (f_i(s) - z_i^min) / a_i,

or, if I store absolute intercept coordinates instead of translated intercept lengths, the equivalent denominator is a_i - z_i^min. The ideal maps to 0 and each axis intercept maps to 1, and the front roughly lands on the simplex sum_i f_i^n = 1 — the same simplex the reference grid lives on. The frames are aligned. And because I recompute the ideal and the intercepts every generation from the current S_t, the normalization tracks the front as it converges and rescales; the reference grid is effectively re-fitted to wherever the population currently is, which is what lets this handle differently-scaled fronts with no tuning.

With everyone normalized, I associate each solution to a reference direction. The instinct is to associate s to its nearest reference *point*, by Euclidean distance. But that's wrong, and seeing why is the crux of the whole design. A reference point defines a *direction* — a ray from the origin — and what I care about is whether a solution serves that direction, regardless of how far out along it the solution sits. A solution very close to the origin and a solution far out along the same ray both serve that ray equally well for diversity; their difference is convergence, which is the rank's job, not the split's. So the right measure is not distance to the point but the perpendicular distance from the solution to the *line*. Decompose s^n along the reference line w: the component along w is the scalar projection (s^n · w)/||w||, the projected vector is that scalar times the unit direction w/||w||, and the perpendicular distance is what's left over,

  d_perp(s, w) = || s^n - ( (s^n · w) / ||w||^2 ) w ||.

Associate s to the reference point whose line minimizes d_perp, and remember both which reference point pi(s) it serves and that minimal distance d(s). This cleanly factors the two goals I've been juggling since the start: position along the ray is convergence (handled by the front rank), perpendicular offset from the ray is which-direction-am-I-serving (handled here). That's the decoupling crowding distance never had.

Now the actual split. I'm filling K vacancies in P_{t+1} from the last front F_l, and I already have, sitting in P_{t+1}, all the members of fronts 1 through l-1, each associated to some reference direction. Let rho_j be the niche count: how many *already-selected* members (from the earlier fronts) are associated to reference point j. This count tells me which directions are already well-served and which are starving. To get a uniform spread, I should pour survivors into the starving directions first. So: find the reference point with the smallest rho_j — the least-represented direction — break ties at random, and try to give it a member from F_l.

What member? Two cases, and they're different on purpose. If rho_j is zero — this direction has *no* representative yet from the earlier fronts — then any member I add becomes its sole ambassador, so I should add the best possible one: the F_l member associated to j with the *shortest* perpendicular distance d_perp, the one that lies most squarely on the ray. If instead rho_j >= 1 — the direction already has a representative — then I just need *a* second body there to keep filling it out; which one barely matters, so I add a *random* associated member. That asymmetry is deliberate: spend the effort of picking the closest only when establishing a brand-new niche; once a niche exists, picking at random is cheaper and avoids greedily over-optimizing a direction that's already covered. After adding a member, increment rho_j and remove it from F_l. If a reference point has no remaining member from F_l, it cannot help this generation; an implementation can exclude it explicitly, or equivalently build the candidate reference set only from niches that still have available F_l members. Repeat the whole pick-emptiest-direction step K times, until P_{t+1} is full. Let me write it in that latter form:

  def niching(F_l, K, niche_of, dist_of, rho):
      selected = []
      available = ones(len(F_l), bool)
      while len(selected) < K:
          # least-represented reference points still in play
          live = unique(niche_of[available])
          jmin = live[ rho[live] == rho[live].min() ]
          shuffle(jmin)
          for j in jmin[: K - len(selected)]:
              members = where((niche_of == j) & available)[0]
              if rho[j] == 0:
                  pick = members[ argmin(dist_of[members]) ]   # closest -> new niche's ambassador
              else:
                  pick = members[0]              # any -> niche already represented (random order)
              available[pick] = False
              rho[j] += 1
              selected.append(F_l[pick])
      return selected

Stepping back, the survival step is now: sort the combined pool into fronts; add whole fronts to S_t until it would overflow; if it lands exactly at N, done; otherwise keep fronts 1..l-1 outright, normalize S_t adaptively (ideal, extremes via ASF, intercepts), associate every member to a reference direction by perpendicular distance, count niches over the already-selected members, and run the niching fill to draw the remaining K from F_l. The reference grid never changes; the normalization re-fits it to the current front each generation; diversity is supplied by the grid and enforced by the emptiest-niche-first fill. Nothing here estimates the population's own density, so nothing degrades as M grows.

That leaves the front edges of the algorithm — mating selection and variation — and I want to be careful not to over-engineer them, because the split is now doing all the diversity work. Mating selection: in NSGA-II I'd use a binary tournament on rank-then-crowding to bias mating toward good, spread-out parents. But crowding distance is the thing I just threw out, and the diversity it would have protected is already protected by the survival niching. If I select survivors uniformly across all reference directions, the survivor population is *already* well-spread, so I don't need mating selection to enforce spread too — I can just pick parents at random from P_{t+1} and let variation explore. There's no fragile diversity knob in mating because diversity is entirely a survival property.

Variation is SBX plus polynomial mutation, the standard real-coded operators, but the recombination difficulty from earlier — distant parents producing offspring distant from both — needs a deliberate response. SBX has a distribution index eta_c that controls how far children spread from parents: large eta_c concentrates children near the parents, small eta_c lets them fly far. In a many-objective population the parents I draw can be widely separated, and if eta_c is small those two distant parents make a child far from each, which lands in an unexplored void and is usually useless. So I want eta_c *large* — keep children close to their parents, which acts like a soft mating restriction, recombining locally rather than blending across the whole spread-out population. So I push eta_c up to 30, larger than the usual 20, specifically to counter that failure mode. Polynomial mutation stays standard: eta_m = 20 and per-variable probability p_m = 1/n so about one of the n variables mutates per individual. SBX crossover probability I take as 1 — always recombine. None of these introduce a diversity parameter; the only structural choices are the reference grid (H, the user's resolution knob, not an algorithmic one) and the population size N tied to H.

Let me also sanity-check the cost, because a big motivation was that crowding got expensive. The non-dominated sort of the 2N pool is O(N log^{M-2} N). The normalization is cheap: ideal is O(MN), extremes O(M^2 N) via the ASF over M weight vectors, one M-by-M solve for the intercepts at O(M^3). Association compares each of up to 2N members against H reference points, O(MNH). Niching is O(N H) worst case over the fill loop. With N roughly equal to H, this is O(N^2 log^{M-2} N) or O(N^2 M), dominated by the sort and the association — same ballpark as NSGA-II, no blowup. Good.

Now I want this in code I'd actually run, and I'll express it as the survival operator over a combined pool, because the framework around it is exactly NSGA-II's. The four pieces — extreme points, intercepts, association, niching — assemble into one selection call:

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
    fn = (F - ideal) / (intercepts - ideal + numpy.finfo(float).eps)   # ideal->0, intercept->1
    fn = numpy.repeat(numpy.expand_dims(fn, axis=1), len(ref_points), axis=1)
    norm = numpy.linalg.norm(ref_points, axis=1)
    # scalar projection along each ref line, times the unit direction = projected vector
    d = numpy.sum(fn * ref_points, axis=2) / norm.reshape(1, -1)
    proj = d[:, :, numpy.newaxis] * ref_points[numpy.newaxis, :, :] / norm[numpy.newaxis, :, numpy.newaxis]
    dist = numpy.linalg.norm(proj - fn, axis=2)         # perpendicular distance to each line
    niches = numpy.argmin(dist, axis=1)                 # closest reference line
    dist = dist[range(niches.shape[0]), niches]
    return niches, dist


def niching(last_front, k, niches, distances, niche_counts):
    """Fill k slots from the last front, always serving the reference point
    with the smallest niche count first."""
    selected = []
    available = numpy.ones(len(last_front), dtype=bool)
    while len(selected) < k:
        live = numpy.zeros(len(niche_counts), dtype=bool)
        live[numpy.unique(niches[available])] = True    # ref points with members still left
        min_count = numpy.min(niche_counts[live])
        chosen_refs = numpy.flatnonzero(live & (niche_counts == min_count))
        numpy.random.shuffle(chosen_refs)
        for j in chosen_refs[: k - len(selected)]:
            members = numpy.flatnonzero((niches == j) & available)
            if niche_counts[j] == 0:
                sel = members[numpy.argmin(distances[members])]   # closest -> new niche
            else:
                numpy.random.shuffle(members)
                sel = members[0]                                  # any -> niche already filled
            available[sel] = False
            niche_counts[j] += 1
            selected.append(last_front[sel])
    return selected


def selNSGA3(individuals, k, ref_points, nd="log", best_point=None,
             worst_point=None, extreme_points=None, return_memory=False):
    """Environmental selection: keep whole fronts, then split the last front
    by reference-point niching."""
    if nd == "standard":
        fronts = sortNondominated(individuals, k)
    elif nd == "log":
        fronts = sortLogNondominated(individuals, k)
    else:
        raise Exception("selNSGA3: invalid non-dominated sorting choice")

    F = numpy.array([ind.fitness.wvalues for f in fronts for ind in f])
    F *= -1.0                                                   # minimization frame

    if best_point is not None and worst_point is not None:
        ideal = numpy.min(numpy.concatenate((F, best_point), axis=0), axis=0)
        worst = numpy.max(numpy.concatenate((F, worst_point), axis=0), axis=0)
    else:
        ideal = numpy.min(F, axis=0)
        worst = numpy.max(F, axis=0)

    extremes = find_extreme_points(F, ideal, extreme_points)
    front_worst = numpy.max(F[:sum(len(f) for f in fronts), :], axis=0)
    intercepts = find_intercepts(extremes, ideal, worst, front_worst)
    niches, dist = associate_to_niche(F, ref_points, ideal, intercepts)

    niche_counts = numpy.zeros(len(ref_points), dtype=numpy.int64)
    idx, cnt = numpy.unique(niches[:-len(fronts[-1])], return_counts=True)
    niche_counts[idx] = cnt

    chosen = list(chain(*fronts[:-1]))                          # fronts 1..l-1 kept outright
    n = k - len(chosen)
    chosen.extend(niching(fronts[-1], n,
                          niches[len(chosen):], dist[len(chosen):], niche_counts))

    if return_memory:
        return chosen, NSGA3Memory(ideal, worst, extremes)
    return chosen
```

And the deployment shell is thin, because everything lives in that survival operator. Mating selection is a random shuffle (diversity is a survival property, not a mating one); variation is SBX with the large eta_c plus polynomial mutation; survival is selNSGA3 over parents-plus-offspring with the precomputed reference grid:

```python
import random
from copy import deepcopy


class CustomMOEA:
    """NSGA-III deployment: NSGA-II framework with reference-point niching
    in place of crowding distance."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None,
                 ref_points=None, ref_divisions=4, ref_scaling=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.bounds = bounds
        self.cx_eta = cx_eta                  # large SBX index -> children near parents
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.ref_points = (ref_points if ref_points is not None
                           else uniform_reference_points(n_obj, ref_divisions, ref_scaling))

    def select(self, population, k):
        sel = [deepcopy(ind) for ind in population]   # no mating selection needed
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
        combined = population + offspring                # elitist (mu + lambda)
        return selNSGA3(combined, self.pop_size, self.ref_points)
```

Let me trace the causal chain one more time. I started with an elitist EA whose convergence comes from Pareto rank and whose diversity comes from crowding distance, and watched both fail as objectives grow: rank loses pressure because nearly everything becomes non-dominated, which throws the entire selection burden onto the last-front split, and crowding distance — a per-axis neighbor-gap density estimate — goes both uninformative and expensive in high dimensions, so the split degenerates. The escape was to stop *estimating* diversity from a sparse cloud and instead *impose* it from a fixed external set of targets. Das and Dennis's weighted-sum critique told me not to get those targets by scalarizing, but their simplex construction handed me exactly the targets I needed: H evenly spaced reference directions, with the option to scale and combine layers when a single grid puts too many points on boundary faces. To compare population members against those directions I had to normalize each generation — ideal point from the survivors, extreme points via the achievement scalarizing function so each axis-extreme is a true corner and not a dominated outlier, axis intercepts from the hyperplane those extremes span — re-fitting the grid to the current front. Association uses perpendicular distance to a reference *line*, not distance to a point, which cleanly factors convergence (position along the ray, left to the rank) from diversity (which ray, the split's job). The split fills the emptiest reference direction first, taking the closest member to seed a brand-new niche and a random member once a niche already exists. Diversity being now entirely a survival property lets me drop mating selection to a random draw, and the recombination-in-high-dimensions failure mode pushes the SBX distribution index up to 30 to keep children near their parents. The whole thing keeps NSGA-II's framework and cost, swaps only the last-front rule, and avoids the high-dimensional density estimate that broke the original split.
