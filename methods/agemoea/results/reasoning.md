Let me start from what actually goes wrong. I have a multi-objective problem: minimize a vector of conflicting objectives, and the answer isn't a point, it's a whole front of non-dominated trade-offs. My job is to return a finite, evenly spread set of points that hug that front. Two things I have to get right at once — convergence, sit as close to the true front as I can, and diversity, spread evenly so the decision-maker sees the whole range of trade-offs instead of a clump. And here's the thing that keeps biting me: both of those are *distances in objective space*, and every distance I write down secretly assumes a shape. Whether two objective vectors are "close" or "well spread apart" depends entirely on the geometry of the front they live on. A flat front, a bulging convex sheet, a spherical cap, a rectangular box — the very same pair of points reads as crowded under one geometry and nicely separated under another.

So let me look hard at what I do today and find exactly where the shape assumption is hiding. NSGA-II is the workhorse. Fast non-dominated sort splits the merged parent-plus-offspring pool into fronts F1, F2, and so on; I fill the next generation front by front, accepting whole fronts until one of them — the critical front — would overflow the population size, and then I trim that critical front by crowding distance. The crowding distance is the part I want to stare at. For each objective m, I sort the front by f_m, hand the two boundary points an infinite distance so the extremes are always kept, and for each interior point I add the normalized neighbor gap (f_m[i+1] − f_m[i−1])/(f_m^max − f_m^min); then I sum those contributions across all M objectives. That sum-over-axes is what I want to interrogate. Adding up per-axis gaps is a Manhattan, L1, cuboid measure of density: it estimates how crowded a point is by drawing a box around it from its axis-aligned neighbors. Let me ask what front that's the *right* density estimate for. On a perfectly flat front — the simplex face where the objectives trade off linearly — the front locally *is* its axis projections, so summing axis gaps measures real arc-length density. On a convex or concave sheet that stops being true: the front curves away from the box, so the axis-projection sum no longer tracks the local density along the actual surface. So crowding distance silently assumes the front is flat. And there's a second hole I keep glossing over: crowding distance only *spreads*. Within a single front it has no opinion about which points are better converged — closer to the ideal corner of the objective space — it just wants them apart. For M up to three I can live with it, but the cuboid estimate degrades as M grows, where most points end up with infinite or near-identical crowding distance and the metric stops discriminating.

Fine, so who else is on the table, and what shape does each of *them* assume? MOEA/D decomposes the problem into N scalar subproblems with uniformly spread weight vectors and a Tchebycheff or PBI scalarizer. The weights are spread uniformly on a flat simplex — so the induced spacing of solutions is calibrated for a near-linear front; on a strongly convex or concave front, uniform weights map to a clumped, uneven distribution of objective vectors. Shape assumption: flat. NSGA-III, built for many objectives, places structured Das–Dennis reference points on the unit simplex and associates each solution to its nearest reference line by perpendicular distance. Same flat hyperplane, plus a perpendicular-Euclidean niching — calibrated for linear fronts, gives up versatility on curved ones. RVEA uses reference vectors and an angle-penalized distance, mixing vector length for convergence with the angle to a reference direction for diversity. Measuring spread by angle implicitly assumes a *spherical* geometry — strong on concave fronts, weaker where angle is the wrong density notion. SPEA2 uses a k-nearest-neighbor density in plain Euclidean distance — Euclidean is again one fixed geometry. Every one of these either hard-codes a single geometry into its diversity-and-convergence metric, or — like the front-modeling line, GFM-MOEA fitting a generalized Lp manifold by Levenberg–Marquardt — pays a heavy fitting cost (something like O(G'·M²·(M+N)) per refit, so it's only re-run every K generations) to learn one. And the empirical pattern that follows is exactly what you'd predict: on benchmark suites built to vary the front shape — WFG with its convex, degenerate, concave, multimodal fronts; ZDT/DTLZ ranging from convex ZDT1 to disconnected ZDT3 to spherical DTLZ2 to linear DTLZ1 — the relative ranking of these algorithms *flips with the geometry*. The reference-point ones win on linear fronts and slide on spherical; the angle ones do the reverse. Nobody is uniformly good, and the reason is uniform: each measures distance with the wrong ruler on the wrong shape.

So the problem crystallizes. I don't want to pick a geometry. I'd like to *measure* the front's geometry from the current population and then use the matching ruler for both diversity and convergence — and I need it cheap enough to redo every generation as the front moves, because GFM's per-K-generation LM fit is exactly the cost I can't afford if I want continuous adaptation. The open question is whether "front geometry" even collapses to something I can estimate cheaply, or whether I'm doomed to the full nonlinear fit.

Let me look for a single handle on it. Consider a *normalized* front — objectives translated so the ideal point sits at the origin and scaled so the front spans [0,1] on each axis. What family of shapes do the benchmark fronts actually trace out? Convex ZDT1, linear DTLZ1, spherical DTLZ2, box-like WFG cases — that list is suspicious, because it's exactly the family swept out by the unit sphere of the Minkowski Lp norm: the set of points with a_1^p + a_2^p + … + a_M^p = 1 in the positive orthant, where a normalized minimization front lives. Watch what p does. At p = 1 that's the flat simplex face Σ a_i = 1, a hyperplane — the linear front. At p = 2 it's the Euclidean unit sphere, the spherical cap. At p < 1 the set bulges *toward* the origin — the convex/hyperbolic front. As p → ∞ it becomes the unit box, the rectangular front. So the whole zoo of front shapes I care about is parameterized by one scalar, the Minkowski exponent p. And there's a bonus that makes this more than a labeling trick: the Lp norm *with that same p* is the natural distance for a front of that shape — L2 for a sphere, L1 for a flat face — because the unit ball of Lp is precisely the region the front bounds. If I could read off p, I'd get my ruler for free: measure both crowding and proximity-to-ideal in the Lp norm with the estimated p, and the metric matches the geometry by construction. That would reframe the whole task into estimating one number p per generation, then running NSGA-II's machinery with Lp-flavored distances instead of the fixed L1 cuboid. The whole plan now hinges on whether p is actually estimable cheaply, so let me chase that down before committing.

Where does environmental selection happen, and what do I need before I can even talk about Lp distances? It happens on the merged pool after non-dominated sorting, when I trim the critical front. But to talk about an Lp *unit* manifold I need the front normalized into [0,1] with the ideal at the origin — otherwise "a_i^p" is meaningless, the objectives have arbitrary scales and offsets. So step zero is normalization, and I'd rather reuse the most robust normalization already known than reinvent it. NSGA-III's normalization is geometry-agnostic, which is what I want: compute the ideal point z^min as the per-axis minimum of the (best, first) non-dominated front, translate it to the origin, then estimate the nadir not by a single solution (which is fragile — no single point need attain the true per-axis maxima) but by fitting the linear hyperplane through M extreme points and reading off its axis intercepts. Let me make the extreme-point detection concrete, because I'll need it. For each axis j, I want the front point that lies closest to that axis — the "corner" pulling out along objective j. Measure the perpendicular distance from a point P to the line through the origin along axis direction B: project P onto B, t = (P·B)/(B·B), and the residual is P − t·B; I only need the argmin, so the squared residual is enough. The point of the front minimizing that for axis direction B = e_j is the j-th corner solution. One numerical nicety I'd better build in: use B = e_j + tiny (say add 1e-6 to every coordinate) rather than the bare axis, so the line isn't exactly degenerate and ties break sanely. Collect the M corner points, stack them as rows of a matrix, and solve [extreme points]·β = 1 for β; the hyperplane through those points hits axis j at intercept 1/β_j, so the normalization vector is the reciprocal of β. If that solve misbehaves — a NaN, an Inf, a non-positive intercept, or duplicate corners — I fall back to per-axis maxima. Then divide the front by the normalization vector and I'm on [0,1] with the ideal at the origin. That's NSGA-III's contribution carried over wholesale; the new question — can I get p — starts now.

Now to estimating p. I have a normalized first front sitting (approximately) on some unknown Lp manifold a_1^p + … + a_M^p = 1, and I want the p that best describes it. The honest thing would be to fit p (and maybe per-axis coefficients) by nonlinear least squares over all the front points — but that's precisely GFM's expensive LM route, O(G'·M²·(M+N)), and the whole reason I'm here is that I refuse to pay it every generation. So I want a one-shot estimate. Do I actually need every point? The manifold equation Σ a_i^p = 1 is one scalar equation in the one unknown p, so a single point on the manifold already over-determines p — any point that genuinely satisfies the equation pins it down. Which point should I trust most? Let me think about which points carry information about p and which don't. A corner point sits at coordinate ≈1 on one axis and ≈0 on the rest, so Σ a_i^p ≈ 1^p + 0 + … = 1 for *any* p — corners tell me nothing about curvature. The curvature shows up where all coordinates are comparable, i.e. near the center of the front, the point closest to the diagonal direction (1, 1, …, 1). So I'll take the front point minimizing perpendicular distance to the diagonal line through the origin and (1,…,1), excluding the corners, and call it C.

Now solve Σ_i C_i^p = 1 for p from that single point. In general that's transcendental. But if C really sits at the symmetric center of the manifold, all its coordinates are about equal to their mean c̄ = (Σ_i C_i)/M, and substituting a_i ≈ c̄ into the manifold equation gives M · c̄^p = 1, which I can solve in closed form: log M + p·log c̄ = 0, so p = −log M / log c̄ = log(M) / log(1 / c̄), where c̄ = mean_i(C_i). One mean, two logs. Before I trust it I want to actually run it on shapes where I know the true p, not just assert it recovers them.

Let me work the cases. Flat front, M = 2: the center of the simplex face a_1+a_2=1 is (0.5, 0.5), c̄ = 0.5, and p = log 2 / log(1/0.5) = log 2 / log 2 = 1. Spherical front, M = 2: the center of the unit quarter-circle is (1/√2, 1/√2), c̄ = 1/√2 ≈ 0.7071, and p = log 2 / log(√2) = log 2 / (½ log 2) = 2. Those two I can do by hand, but I want to be sure I haven't only checked the textbook endpoints, so let me push it numerically including a genuinely non-trivial exponent and higher M. Running p = log(M)/log(1/mean(C)) on the exact centers:

```
flat   M=2  C=(0.5,0.5)              -> p = 1.0000
flat   M=3  C=(1/3,1/3,1/3)          -> p = 1.0000
sphere M=2  C=(0.7071,0.7071)        -> p = 2.0000
sphere M=3  C=(0.5774,0.5774,0.5774) -> p = 2.0000
convex p=0.5 M=2  C=(0.25,0.25)      -> p = 0.5000
```

So it nails the flat and spherical cases at both M=2 and M=3, and — the check I actually cared about — it recovers p = 0.5 on the convex front a_1^0.5 + a_2^0.5 = 1, whose center solves c̄^0.5 = 0.5, i.e. c̄ = 0.25, giving log 2 / log 4 = 0.5 exactly. That's neither endpoint, so the formula isn't just memorizing the two famous shapes; it's reading the exponent off the geometry.

But that all assumed C sits *exactly* at the symmetric center, and in a real population it won't — the nearest point to the diagonal is only approximately central. So how badly does the symmetric-center assumption bite when C drifts off-diagonal? Let me take true points on the sphere (true p = 2) at decreasing angle from the diagonal and feed each through the formula:

```
point on circle, true p=2:
  45 deg  (0.707,0.707)  mean=0.7071  -> p_est = 2.000
  40 deg  (0.766,0.643)  mean=0.7044  -> p_est = 1.978
  35 deg  (0.819,0.574)  mean=0.6964  -> p_est = 1.915
  30 deg  (0.866,0.500)  mean=0.6830  -> p_est = 1.818
```

So the estimate is exact only on the diagonal and drifts low as the point moves toward a corner — at 30° it reports 1.82 instead of 2.0. And I can see why directly: plugging the 35° estimate p=1.915 back into the true manifold equation gives a_1^p + a_2^p = 1.027, not 1, so the symmetric substitution is the source of the error, not anything else. That's not a defect to patch — it's the justification for the choice I already made: I pick the point *nearest the diagonal* precisely because that's where the symmetric assumption is least violated and the estimate is most accurate. If I'd picked an arbitrary front point the bias would be much worse. Good — the design of "central point, excluding corners" is doing real work, and I've now seen the magnitude of the error it controls. I should still guard the formula where it can blow up: if the estimate is NaN or ≤ 0.1, fall back to p = 1; if p is larger than 20, cap it at 20 to avoid |a|^p underflow. With that, I can re-estimate the front's geometry every generation without an iterative nonlinear fit — which is the continuous adaptation GFM couldn't afford.

With p in hand I can build the survival metric, and I want it to fix *both* of NSGA-II's holes: measure crowding in the right geometry, and reward convergence within a front, which plain crowding distance ignores. Take convergence first because it's the simpler of the two. After normalization the ideal point of the first front is the origin, so a point A that is well-converged is one *close to the origin* — and "close" should be measured in the front's own geometry, i.e. by the Lp norm ||A||_p = (Σ_i |A_i|^p)^(1/p). Smaller is better-converged. For fronts after the first one, I do not estimate a fresh shape and I do not compute neighbor diversity; I reuse the first front's normalization and p, divide those objective vectors by the same normalization vector, and assign the inverse Minkowski distance to the first front's ideal point. That is a proximity-only score for already-ranked deeper fronts, and it keeps the whole population under the same ruler.

Now diversity on the critical front, in the Lp geometry. The NSGA-II instinct — a point is "diverse" if it's far from its neighbors — carries over; I just measure that distance in Lp, not as the L1 cuboid. So the diversity of A within a front is min over the other points B of ||A − B||_p, its Lp distance to its nearest neighbor: an isolated point has a large nearest-neighbor distance and should be kept; a point with a near-duplicate sitting on top of it has a tiny one and is expendable. But I shouldn't apply this as a one-shot static ranking, because trimming a front is inherently sequential — once I keep some points, the "neighbors" of the remaining candidates are the *already-kept* set, not the original front. So I'll select greedily. Start by keeping the M corner points unconditionally — they define the spread of the front and the normalization, exactly the role NSGA-II's boundary points play with their infinite crowding distance — and mark them selected with score ∞. Then repeatedly: among the not-yet-selected points, for each candidate look at its Lp distances to the currently selected set, take the two smallest of those (the candidate's two nearest selected neighbors), and sum them; pick the candidate whose summed distance to its two nearest selected neighbors is *largest* — sitting in the biggest remaining gap — add it to the selected set, and record that sum as its survival score. Iterate until the front is exhausted. Using the two nearest selected neighbors rather than just the single nearest is the same flavor as NSGA-II using both flanking neighbors per axis: it makes the gap measure two-sided and more stable, less fooled by a single close point.

I'm now computing two separate things — a proximity score (1/||A||_p) and a diversity score (greedy nearest-selected-neighbor Lp gaps) — and I want them to act together in one survival ranking on the critical front, not as two lists I have to reconcile by hand. Here's the move: when I compute the pairwise Lp distances on the critical front, divide each candidate A's row of distances by A's own proximity norm ||A||_p before running the greedy gap selection. So the "distance" the greedy selector sees from candidate A is its raw Lp gap to neighbors *scaled by 1/||A||_p*. Let me reason through what that does to the four corners of the trade-off. A point that is far from its neighbors (good spread) *and* close to the ideal (small ||A||_p, good convergence) has its already-large gaps divided by a small number — boosted twice. A point that is isolated but poorly converged (large ||A||_p) has its gaps shrunk, favored less. A point near the ideal but crowded still has tiny gaps and loses. So the single scaled-gap score blends diversity and proximity multiplicatively, both in the estimated Lp geometry, in one greedy pass.

Before I believe that, let me actually trace the greedy fill on a tiny front and check the deferral behavior I'm claiming — that a near-duplicate gets kept *last*. Take a flat front (p=1, M=2) of five normalized points: two corners (0,1) and (1,0), the center (0.5,0.5), an interior point (0.24,0.76), and a near-duplicate of it (0.26,0.74) forming a clump. Keep the corners first (score ∞), then run the scaled-gap greedy fill. The selection order it produces:

```
1st kept after corners:  (0.50,0.50)   score 2.00
2nd:                     (0.24,0.76)   score 1.00
3rd (last):              (0.26,0.74)   score 0.52
final scores: [inf, inf, 2.0, 1.0, 0.52]
```

That's the behavior I wanted, computed rather than hoped: the center, sitting in the largest empty gap, is taken first; the well-separated interior point next; and the near-duplicate of the clump is taken *last* with the smallest finite score (0.52), so if this front were the critical one being trimmed, the near-duplicate is exactly what gets dropped first. The greedy max-min-gap fill builds an evenly spread subset rather than deleting near-duplicates blindly, and the scaled-gap score gives me a single ordering I can just sort by. That's the survival score. Trim the critical front by keeping the points with the highest survival scores until the population is full.

Let me also pin down the degenerate guards before I lose them. If a point coincides with the ideal so ||A||_p is ~0, dividing by it explodes; floor those norms by replacing anything below 1e-8 with 1. Floor pairwise distances below 1e-8 to 1e-8 before the greedy selection. If the first front has fewer points than objectives, the corner set and hyperplane are underdetermined, so return zero scores for that front, use p = 1, and use the per-axis maximum of that raw front as the normalization.

Let me trace the whole environmental selection once end to end to be sure the pieces connect. Merge parents and offspring into a pool of size 2N. Non-dominated sort into F1, F2, …. Take F1, the best front; from it compute the ideal point (per-axis min), translate to origin, find the M corner points by the perpendicular-to-axis distance, solve for the hyperplane intercepts to get the normalization, scale F1 to [0,1]. On the normalized F1 find the central point (closest to the diagonal, excluding corners) and compute p = log(M)/log(1/mean(central)). Compute F1's survival scores: corners get ∞; everyone else gets the greedy scaled-gap score in the Lp norm. For each deeper front F_d, divide its raw objective vectors by the same normalization and score by inverse Lp distance to the first-front ideal point using the same p. Now do the elitist fill: accept whole fronts in order while they fit; for the critical front that overflows, sort its members by survival score descending and keep exactly as many as needed to reach N. The points kept become the next generation. Mating selection is then a binary tournament: prefer lower front rank, break ties by higher survival score — the geometry-aware analogue of NSGA-II's crowded-comparison operator. Variation stays the standard real-coded pair, SBX crossover and polynomial mutation with p_m = 1/n per variable; those aren't where the contribution is, and changing them would only muddy the comparison, so I keep them exactly as the baselines use them. The whole novelty lives in `survive`: no LM fit, no per-K-generation lag, just first-front normalization and a fresh scalar exponent feeding the front-distance computations already needed for selection.

Let me write it as the code I'd actually run, filling the one empty slot — the environmental-selection metric — in the generational MOEA harness, with the mating tournament and the standard operators around it.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # fast non-dominated sort, SBX, polynomial mutation


class CustomMOEA:
    """Adaptive-geometry MOEA: estimate the Pareto front's Minkowski exponent p each
    generation and trim the critical front by an L_p survival score that fuses
    proximity-to-ideal (convergence) with nearest-neighbor spread (diversity)."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                      # (low, up) arrays
        self.cx_eta = cx_eta                      # SBX distribution index
        self.mut_eta = mut_eta                    # polynomial-mutation distribution index
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    # ---- mating selection: binary tournament by (rank, then survival score) ----
    def select(self, population, k):
        fronts = tools.sortNondominated(population, len(population))
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank                  # lower rank = better front
        selected = []
        for _ in range(k):
            a, b = random.sample(population, 2)
            if a._rank < b._rank:
                selected.append(deepcopy(a))
            elif b._rank < a._rank:
                selected.append(deepcopy(b))
            else:                                 # tie on rank -> keep higher survival score
                sa = getattr(a, "_score", 0.0)
                sb = getattr(b, "_score", 0.0)
                winner = a if sa >= sb else b
                selected.append(deepcopy(winner))
        return selected

    # ---- variation: standard SBX crossover + polynomial mutation ----
    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1], eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    # ---- environmental selection (the contribution) ----
    def survive(self, population, offspring):
        combined = population + offspring
        F = np.array([ind.fitness.values for ind in combined], dtype=float)
        fronts = tools.sortNondominated(combined, len(combined))

        # rank every individual; score buffer for the whole pool
        score = np.zeros(len(combined))
        index_of = {id(ind): i for i, ind in enumerate(combined)}
        for rank, front in enumerate(fronts):
            for ind in front:
                ind._rank = rank

        # ---- geometry from the FIRST (best) front: ideal, normalization, p ----
        f1_idx = [index_of[id(ind)] for ind in fronts[0]]
        front1 = F[f1_idx]
        ideal = np.min(front1, axis=0)                       # ideal point = per-axis min
        s1, p, normalization = self._survival_score(front1, ideal)
        for local, gi in enumerate(f1_idx):
            score[gi] = s1[local]

        # deeper fronts: proximity-only score, SAME normalization and SAME p
        for rank in range(1, len(fronts)):
            idx = [index_of[id(ind)] for ind in fronts[rank]]
            fr = F[idx] / normalization
            d = self._minkowski_distances(fr, ideal[None, :], p=p).squeeze()
            score[idx] = 1.0 / d

        for ind in combined:                                  # expose for the tournament
            ind._score = score[index_of[id(ind)]]

        # ---- elitist fill: whole fronts, then trim the critical one by survival score ----
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                if remaining <= 0:
                    break
                ranked = sorted(front, key=lambda ind: ind._score, reverse=True)
                next_gen.extend(ranked[:remaining])
                break
        return next_gen

    def _survival_score(self, front, ideal):
        front = np.round(front, 12)
        m, n = front.shape
        scores = np.zeros(m)
        if m < n:                                             # too few points to fit geometry
            return scores, 1.0, np.max(front, axis=0)

        front = front - ideal                                 # shift ideal to origin
        extreme = self._find_corners(front)                   # M corner solutions
        front, normalization = self._normalize(front, extreme)

        scores[extreme] = np.inf                              # keep the corners (spread)
        selected = np.zeros(m, dtype=bool)
        selected[extreme] = True

        p = self._estimate_p(front, extreme, n)               # Eq.: log(M)/log(1/mean(C))

        nn = np.linalg.norm(front, p, axis=1)                 # proximity ||A||_p per point
        nn[nn < 1e-8] = 1.0
        dist = self._pairwise_lp(front, p)                    # L_p gaps between points
        dist[dist < 1e-8] = 1e-8
        dist = dist / nn[:, None]                             # scale gaps by 1/||A||_p

        neighbors = 2
        remaining = list(np.where(~selected)[0])
        for _ in range(m - int(selected.sum())):
            sel_idx = np.where(selected)[0]
            D = dist[np.ix_(remaining, sel_idx)]              # candidates x already-kept
            if D.shape[1] > 1:                                # sum of 2 nearest kept neighbors
                k = min(neighbors, D.shape[1])
                part = np.partition(D, k - 1, axis=1)[:, :k]
                gap = np.sum(part, axis=1)
            else:
                gap = D[:, 0]
            best = int(np.argmax(gap))                        # biggest remaining gap
            gi = remaining.pop(best)
            selected[gi] = True
            scores[gi] = gap[best]
        return scores, p, normalization

    @staticmethod
    def _estimate_p(front, extreme, n):
        d = CustomMOEA._point_to_line(front, np.zeros(n), np.ones(n))  # dist to diagonal
        d[extreme] = np.inf                                   # central point = nearest diagonal
        c = front[int(np.argmin(d))]
        mean_c = np.mean(c)
        p = np.log(n) / np.log(1.0 / mean_c)                  # M c_bar^p = 1  =>  this p
        if np.isnan(p) or p <= 0.1:
            return 1.0
        return min(p, 20.0)                                   # cap to avoid |x|^p underflow

    @staticmethod
    def _find_corners(front):
        m, n = front.shape
        if m <= n:
            return np.arange(m)
        W = 1e-6 + np.eye(n)                                  # axes, slightly perturbed
        idx = np.zeros(n, dtype=int)
        taken = np.zeros(m, dtype=bool)
        for i in range(n):
            d = CustomMOEA._point_to_line(front, np.zeros(n), W[i])
            d[taken] = np.inf
            j = int(np.argmin(d))                             # point closest to axis i
            idx[i] = j
            taken[j] = True
        return idx

    @staticmethod
    def _point_to_line(P, A, B):
        ba = B - A
        d = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            pa = P[i] - A
            t = np.dot(pa, ba) / np.dot(ba, ba)               # project pa onto ba
            d[i] = np.sum((pa - t * ba) ** 2)                 # squared perpendicular distance
        return d

    @staticmethod
    def _normalize(front, extreme):
        m, n = front.shape
        if len(np.unique(extreme)) != len(extreme):           # duplicate corners -> fallback
            norm = np.max(front, axis=0)
            return front / norm, norm
        try:                                                  # hyperplane through corners
            beta = np.linalg.solve(front[extreme], np.ones(n))
            if np.any(np.isnan(beta)) or np.any(np.isinf(beta)) or np.any(beta <= 0):
                norm = np.max(front, axis=0)
            else:
                norm = 1.0 / beta                             # axis intercepts of the plane
                if np.any(np.isnan(norm)) or np.any(np.isinf(norm)):
                    norm = np.max(front, axis=0)
        except np.linalg.LinAlgError:
            norm = np.max(front, axis=0)
        norm[norm == 0.0] = 1.0
        return front / norm, norm

    @staticmethod
    def _pairwise_lp(front, p):
        m = front.shape[0]
        dist = np.zeros((m, m))
        for i in range(m):
            dist[i] = np.sum(np.abs(front[i] - front) ** p, axis=1) ** (1.0 / p)
        return dist

    @staticmethod
    def _minkowski_distances(A, B, p):
        m1 = A.shape[0]
        m2 = B.shape[0]
        dist = np.zeros((m1, m2))
        for i in range(m1):
            for j in range(m2):
                dist[i, j] = np.sum(np.abs(A[i] - B[j]) ** p) ** (1.0 / p)
        return dist

    def on_generation(self, gen, population):
        pass
```
