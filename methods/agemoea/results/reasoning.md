Let me start from what actually goes wrong. I have a multi-objective problem: minimize a vector of conflicting objectives, and the answer isn't a point, it's a whole front of non-dominated trade-offs. My job is to return a finite, evenly spread set of points that hug that front. Two things I have to get right at once — convergence, sit as close to the true front as I can, and diversity, spread evenly so the decision-maker sees the whole range of trade-offs instead of a clump. And here's the thing that keeps biting me: both of those are *distances in objective space*, and every distance I write down secretly assumes a shape. Whether two objective vectors are "close" or "well spread apart" depends entirely on the geometry of the front they live on. A flat front, a bulging convex sheet, a spherical cap, a rectangular box — the very same pair of points reads as crowded under one geometry and nicely separated under another.

So let me look hard at what I do today and find exactly where the shape assumption is hiding. NSGA-II is the workhorse. Fast non-dominated sort splits the merged parent-plus-offspring pool into fronts F1, F2, and so on; I fill the next generation front by front, accepting whole fronts until one of them — the critical front — would overflow the population size, and then I trim that critical front by crowding distance. The crowding distance is the part I want to stare at. For each objective m, I sort the front by f_m, hand the two boundary points an infinite distance so the extremes are always kept, and for each interior point I add the normalized neighbor gap (f_m[i+1] − f_m[i−1])/(f_m^max − f_m^min); then I sum those contributions across all M objectives. That sum-over-axes is the tell. Adding up per-axis gaps is a Manhattan, L1, cuboid measure of density. It estimates how crowded a point is by drawing a box around it from its axis-aligned neighbors and treating the front as if it were flat. On a flat front that's fine. On a convex or concave front it mis-reads who is actually crowded, because the real local density along a curved sheet isn't the sum of axis projections. And there's a second hole I keep glossing over: crowding distance only *spreads*. Within a single front it has no opinion about which points are better converged — closer to the ideal corner of the objective space — it just wants them apart. For M up to three I can live with it, but the cuboid estimate falls apart as M grows, where most points end up with infinite or near-identical crowding distance and the metric stops discriminating at all.

Fine, so who else is on the table, and what shape does each of *them* assume? MOEA/D decomposes the problem into N scalar subproblems with uniformly spread weight vectors and a Tchebycheff or PBI scalarizer. The weights are spread uniformly on a flat simplex — so the induced spacing of solutions is calibrated for a near-linear front; on a strongly convex or concave front, uniform weights map to a clumped, uneven distribution of objective vectors. Shape assumption: flat. NSGA-III, built for many objectives, places structured Das–Dennis reference points on the unit simplex and associates each solution to its nearest reference line by perpendicular distance. Same flat hyperplane, plus a perpendicular-Euclidean niching — calibrated for linear fronts, gives up versatility on curved ones. RVEA uses reference vectors and an angle-penalized distance, mixing vector length for convergence with the angle to a reference direction for diversity. Measuring spread by angle implicitly assumes a *spherical* geometry — strong on concave fronts, weaker where angle is the wrong density notion. SPEA2 uses a k-nearest-neighbor density in plain Euclidean distance — Euclidean is again one fixed geometry. Every one of these either hard-codes a single geometry into its diversity-and-convergence metric, or — like the front-modeling line, GFM-MOEA fitting a generalized Lp manifold by Levenberg–Marquardt — pays a heavy fitting cost (something like O(G'·M²·(M+N)) per refit, so it's only re-run every K generations) to learn one. And the empirical pattern that follows is exactly what you'd predict: on benchmark suites built to vary the front shape — WFG with its convex, degenerate, concave, multimodal fronts; ZDT/DTLZ ranging from convex ZDT1 to disconnected ZDT3 to spherical DTLZ2 to linear DTLZ1 — the relative ranking of these algorithms *flips with the geometry*. The reference-point ones win on linear fronts and slide on spherical; the angle ones do the reverse. Nobody is uniformly good, and the reason is uniform: each measures distance with the wrong ruler on the wrong shape.

So the problem crystallizes. I don't want to pick a geometry. I want to *measure* the front's geometry from the current population and then use the matching ruler for both diversity and convergence — and I need it cheap enough to redo every generation as the front moves, because GFM's per-K-generation LM fit is exactly the cost I can't afford if I want continuous adaptation.

Now, do I have a single handle on "front geometry"? Let me think about a *normalized* front — objectives translated so the ideal point sits at the origin and scaled so the front spans [0,1] on each axis. The clean family of shapes is the unit sphere of the Minkowski Lp norm: the set of points with a_1^p + a_2^p + … + a_M^p = 1 (taking the points in the positive orthant, where a normalized minimization front lives). Watch what p does. At p = 1 that's the flat simplex face Σ a_i = 1, a hyperplane — the linear front. At p = 2 it's the Euclidean unit sphere, the spherical cap. At p < 1 the set bulges *toward* the origin — the convex/hyperbolic front. As p → ∞ it becomes the unit box, the rectangular front. So the whole zoo of front shapes I care about is one scalar, the Minkowski exponent p, and — this is the part that makes it useful — the Lp norm *with that same p* is the natural distance for a front of that shape. L2 for a sphere, L1 for a flat face. If I can read off p, I get my ruler for free: measure both crowding and proximity-to-ideal in the Lp norm with the estimated p, and the metric automatically matches the geometry. That reframes the whole task: estimate one number p per generation, then run NSGA-II's machinery but with Lp-flavored distances instead of the fixed L1 cuboid.

Where does environmental selection happen, and what do I need before I can even talk about Lp distances? It happens on the merged pool after non-dominated sorting, when I trim the critical front. But to talk about an Lp *unit* manifold I need the front normalized into [0,1] with the ideal at the origin — otherwise "a_i^p" is meaningless, the objectives have arbitrary scales and offsets. So step zero is normalization, and I should reuse the most robust normalization already known rather than reinvent it. NSGA-III's normalization is exactly right and geometry-agnostic: compute the ideal point z^min as the per-axis minimum of the (best, first) non-dominated front, translate it to the origin, then estimate the nadir not by a single solution (which is fragile — no single point need attain the true per-axis maxima) but by fitting the linear hyperplane through M extreme points and reading off its axis intercepts. Let me make the extreme-point detection concrete, because I'll need it. For each axis j, I want the front point that lies closest to that axis — the "corner" pulling out along objective j. Measure the perpendicular distance from a point P to the line through the origin along axis direction B: project P onto B, t = (P·B)/(B·B), and the residual is P − t·B; I only need the argmin, so the squared residual is enough. The point of the front minimizing that for axis direction B = e_j is the j-th extreme/corner solution. One numerical nicety I'd better build in: use B = e_j + tiny (say add 1e-6 to every coordinate) rather than the bare axis, so the line isn't exactly degenerate and ties break sanely. Collect the M corner points, stack them as rows of a matrix, and solve [extreme points]·β = 1 for β; the hyperplane through those points hits axis j at intercept 1/β_j, so the normalization vector is the reciprocal of β. If that solve misbehaves — a NaN, an Inf, a non-positive intercept, or duplicate corners — I fall back to per-axis maxima. Then divide the front by the normalization vector and I'm on [0,1] with the ideal at the origin. Good. That's NSGA-III's contribution carried over wholesale; the new thinking starts now.

Now estimate p. I have a normalized first front sitting (approximately) on some unknown Lp manifold a_1^p + … + a_M^p = 1, and I want the p that best describes it. The honest thing would be to fit p (and maybe per-axis coefficients) by nonlinear least squares over all the front points — but that's precisely GFM's expensive LM route, O(G'·M²·(M+N)), and the whole reason I'm here is that I refuse to pay it every generation. So I want a one-shot estimate. I don't need every point to pin down a single exponent; *one well-chosen point on the manifold already determines p*, because the manifold equation Σ a_i^p = 1 is one scalar equation in the one unknown p. Which point? The most informative one for the exponent is the point near the *center* of the front, the one closest to the diagonal direction (1, 1, …, 1) — the bisector of the orthant — because that's where the curvature of the manifold shows up most cleanly and symmetrically (the extreme/corner points all sit at coordinate ~1 on one axis and ~0 elsewhere, so they satisfy Σ a_i^p ≈ 1 for *any* p and tell me nothing about curvature; the center point is where p actually bites). So: take the front point minimizing the perpendicular distance to the diagonal line through origin and (1,…,1), excluding the extreme points, and call it C, the central solution.

Now solve Σ_i C_i^p = 1 for p from that single point. In general that's still a transcendental equation, but if C really sits at the symmetric center of the manifold, all its coordinates are about equal to their mean, call it c̄ = (Σ_i C_i)/M. Substitute the symmetric assumption a_i ≈ c̄ into the manifold equation: M · c̄^p = 1. That's now trivially solvable. Take logs: log M + p·log c̄ = 0, so p = −log M / log c̄ = log M / log(1/c̄) = log(M) / log(1 / mean_i(C_i)). There's my closed-form estimate of the geometry — one mean, two logs after I have chosen the central point. Let me sanity-check the limits. A flat front: the central point of the simplex face Σ a_i = 1 has all coordinates 1/M, so c̄ = 1/M, and p = log M / log(1/(1/M)) = log M / log M = 1. Flat gives p = 1, exactly right. A spherical front, M = 2: the central point of the unit quarter-circle is (1/√2, 1/√2), so c̄ = 1/√2 and p = log 2 / log(√2) = log 2 / (½ log 2) = 2. Spherical gives p = 2. The estimator recovers the textbook exponents on the textbook shapes, from a single point. And I should guard it exactly where the formula can stop being useful: if the estimate is NaN or ≤ 0.1, fall back to p = 1; if p is larger than 20, cap it at 20 to avoid |a|^p underflow. So I now re-estimate the front's geometry every generation without an iterative nonlinear fit, which is exactly the continuous adaptation GFM couldn't afford.

With p in hand I can finally build the survival metric, and I want it to fix *both* of NSGA-II's holes: measure crowding in the right geometry, and also reward convergence within a front, which plain crowding distance ignores. Take convergence first because it's the simpler of the two. After normalization the ideal point of the first front is the origin, so a point A that is well-converged is one *close to the origin* — and "close" should be measured in the front's own geometry, i.e. by the Lp norm ||A||_p = (Σ_i |A_i|^p)^(1/p). Smaller is better-converged. For fronts after the first one, I do not estimate a fresh shape and I do not compute neighbor diversity; I reuse the first front's normalization and p, divide those objective vectors by the same normalization vector, and assign the inverse Minkowski distance to the first front's ideal point. That is a proximity-only score for already-ranked deeper fronts, and it keeps the whole population under the same ruler.

Now diversity on the critical front, in the Lp geometry. The NSGA-II instinct is right — a point is "diverse" if it's far from its neighbors — but I'll measure that distance in Lp, not as the L1 cuboid. So the diversity of A within a front is min over the other points B of ||A − B||_p, its Lp distance to its nearest neighbor: an isolated point has a large nearest-neighbor distance and should be kept; a point with a near-duplicate sitting on top of it has a tiny one and is expendable. But I don't want to apply this as a one-shot static ranking, because trimming a front is inherently sequential — once I keep some points, the "neighbors" of the remaining candidates are the *already-kept* set, not the original front. So I'll select greedily. Start by keeping the M extreme/corner points unconditionally — they define the spread of the front and the normalization, exactly the role NSGA-II's boundary points play with their infinite crowding distance — and mark them selected with score ∞. Then repeatedly: among the not-yet-selected points, for each candidate look at its Lp distances to the currently selected set, take the two smallest of those (the candidate's two nearest selected neighbors), and sum them; pick the candidate whose summed distance to its two nearest selected neighbors is *largest* — that's the one sitting in the biggest remaining gap — add it to the selected set, and record that sum as its survival score. Iterate until the front is exhausted. Using the two nearest selected neighbors rather than just the single nearest is the same flavor as NSGA-II using both flanking neighbors per axis: it makes the gap measure two-sided and more stable, less fooled by a single close point. The greedy max-min-gap fill is what builds an evenly spread subset rather than just deleting near-duplicates blindly.

But I'm now computing two separate things — a proximity score (1/||A||_p) and a diversity score (greedy nearest-selected-neighbor Lp gaps) — and I want them to act together in a single survival ranking on the critical front. When I compute the pairwise Lp distances on the critical front, I divide each candidate A's row of distances by A's own proximity norm ||A||_p before running the greedy gap selection. So the "distance" the greedy selector sees from candidate A is its raw Lp gap to neighbors *scaled by 1/||A||_p*. Think about what that does. A point that is far from its neighbors (good spread) *and* close to the ideal (small ||A||_p, good convergence) gets its already-large gaps divided by a small number — its score is boosted twice. A point that is isolated but poorly converged (large ||A||_p) has its gaps shrunk, so it's favored less. A point near the ideal but crowded still has tiny gaps and loses. The single scaled-gap score therefore blends diversity and proximity multiplicatively, both measured in the estimated Lp geometry, with one greedy pass. That's the survival score. Trim the critical front by keeping the points with the highest survival scores until the population is full.

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

So the causal chain, start to finish. I was stuck because convergence and diversity are distances and every distance assumes a front shape, and each existing MOEA hard-codes one — NSGA-II's crowding distance is an L1 cuboid that mis-reads density on curved fronts and ignores convergence within a front; MOEA/D and NSGA-III bake in a flat hyperplane; RVEA bakes in spherical angles; GFM-MOEA learns the shape but only by an expensive LM fit it can't afford every generation — so the field's rankings flip with the front geometry. The important observation was that a normalized front is an Lp unit manifold whose single exponent p encodes the shape (1 flat, 2 spherical, <1 convex, →∞ box) and whose matching Lp norm is the correct ruler. To get p cheaply I reused NSGA-III's robust extreme-point-hyperplane normalization to put the first front on the common scale, then noticed that one central point — the one nearest the diagonal — already determines p through Σ C_i^p = 1, which under the symmetric center assumption M·c̄^p = 1 collapses to the closed form p = log(M)/log(1/mean(C)). With p I built a single survival score that fuses both qualities in the estimated geometry: proximity is 1/||A||_p, diversity is a greedy nearest-selected-neighbor gap fill seeded by the kept corners, and dividing each candidate's Lp neighbor gaps by its own ||A||_p fuses spread and convergence into one ranking. Trim the critical front by that score, rank deeper fronts by proximity under the same p and normalization, run the standard SBX/polynomial-mutation variation and a rank-then-score binary tournament around it, and the algorithm re-measures its front model every generation without an iterative fit.
