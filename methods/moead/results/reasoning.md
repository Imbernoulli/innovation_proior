Let me start from what actually hurts when I try to approximate a Pareto front with an evolutionary algorithm. I have `m` objectives `f_1, ..., f_m` that conflict, so there is no single best `x`; the answer is a set — the non-dominated points, where no other feasible point is at least as good on every objective and strictly better on one. The decision maker does not want the whole, possibly infinite, front; they want a finite handful of representative trade-offs, close to the true front and evenly spread along it, out of one run. So my real target is a population that does two things at once: pushes toward the front (convergence) and stays spread out across it (diversity). And the tool everyone reaches for, including me, is dominance.

The trouble with dominance is that it is only a partial order. Take two candidate solutions; if neither dominates the other, dominance has nothing to say about which is better. Early in a run that is fine, lots of points are clearly bad and get dominated away. But watch what happens as the population improves: more and more of it becomes mutually non-dominated, and then dominance falls silent — it cannot separate the members of the first front from each other. NSGA-II, the method I'd otherwise just use, feels this directly. It takes parents `P_t` of size `N`, breeds offspring `Q_t` with SBX and polynomial mutation, pools `R_t = P_t ∪ Q_t` of size `2N`, runs fast non-dominated sorting into fronts `F_1, F_2, ...` (peel off the non-dominated set, peel again, and so on), then fills the next generation front by front. The front that doesn't fit gets truncated by crowding distance: sort each objective, give the two extremes infinite distance, and give each interior point the summed normalized gap to its neighbors, `sum_i (f_i^{next} - f_i^{prev})/(f_i^{max} - f_i^{min})`, keeping the most isolated. It works, and it's the state of the art. But two things gnaw at me. First, once everyone is in `F_1`, the *only* thing deciding survival is that crowding-distance tie-break — a separate density heuristic bolted on for diversity, doing the job that selection pressure was supposed to do. Diversity isn't intrinsic to the method; it's an appliance plugged into the side. Second, that fast non-dominated sort is `O(m N^2)` every generation, and with three or more objectives the situation gets worse on both counts — almost every pair is mutually non-dominated, so dominance carries essentially no information at all, and the sort is just as expensive. SPEA2 and PAES are in the same family: rank by dominance, then add a density device (strength + `k`-NN, or an adaptive grid). Same partial-order weakening, same bolt-on diversity.

So I want to step back and ask: what would give me a *total* order, so selection never goes silent, and would make diversity fall out for free instead of being stapled on? There's a whole mature body of theory in classical, non-evolutionary multi-objective optimization — scalarization — and the dominance-based MOEAs throw all of it away, because they treat the MOP as one indivisible blob and never tie any individual to a scalar problem. Scalarization says: under mild conditions, a Pareto-optimal solution is the optimizer of *some* single scalar function that aggregates all the objectives. If that's true, then approximating the front isn't one hard vector problem; it's a *family* of scalar problems, one per choice of how I weight the objectives. And a scalar problem has a total order — I can always say which of two solutions is better for it. That's exactly the selection pressure dominance lacks. Let me chase this.

A weight vector `lambda = (lambda_1, ..., lambda_m)`, `lambda_i >= 0`, summing to one, fixes a trade-off preference. The obvious scalarization is the weighted sum, `g^ws(x | lambda) = sum_i lambda_i f_i(x)`. Minimize it for a given `lambda` and I land on some Pareto point; sweep `lambda` over the simplex and I should sweep the front. Let me check it actually traces the *whole* front, though, because that's the only thing that matters. Picture the level sets of `g^ws`: they're hyperplanes with normal `lambda`. Minimizing means sliding that hyperplane down until it last touches the attainable objective set. If the front is convex, every point on it is a last-touch point of some supporting hyperplane — fine, every Pareto point is reachable by some `lambda`. But if the front has a re-entrant, nonconvex stretch, a hyperplane sliding in will skip right over it and touch only the two convex shoulders on either side; the points in the dent are never the minimizer of any weighted sum. So weighted sum reaches only the supported Pareto points exposed on the boundary of the convex hull of the attainable objective set. The ZDT and DTLZ suites are full of nonconvex and disconnected fronts — ZDT2 is nonconvex, ZDT3 is in pieces — so weighted sum alone will systematically miss whole regions. Wall. It has the total order I wanted, but it can't see the whole front.

The fix is to change the geometry of the level sets so they can poke into the dents. The Tchebycheff (Chebyshev) scalarization does this: `g^te(x | lambda, z*) = max_i { lambda_i |f_i(x) - z*_i| }`, where `z* = (z*_1, ..., z*_m)` is the ideal point, `z*_i = min_{x in Omega} f_i(x)`. Now the level sets aren't hyperplanes; they're nested axis-aligned right-angle "corners" (rescaled `L_infinity` balls) with their vertex marching out along the ray from `z*` in the direction `1/lambda`. Minimizing `g^te` pulls that corner toward `z*` until its vertex just touches the attainable set. Because the contour is a corner, not a flat plane, it can reach *into* a nonconvex dent and touch a point there that no flat hyperplane could. I need the actual optimality argument, not just the picture. Take a Pareto-optimal `x*` and write the gaps as `d_i = f_i(x*) - z*_i`. If every `d_i > 0`, align the corner with `x*` by choosing `lambda_i` proportional to `1/d_i`; after normalization, every product `lambda_i d_i` equals the same constant `c`. Then `g^te(x* | lambda, z*) = c`. If some feasible `y` had `g^te(y | lambda, z*) < c`, every term would satisfy `lambda_i(f_i(y) - z*_i) < c = lambda_i(f_i(x*) - z*_i)`, hence `f_i(y) < f_i(x*)` for every objective, which would dominate `x*`. That cannot happen. So `x*` is a minimizer. If a gap is zero, the reciprocal construction is not defined; choose one zero-gap coordinate `r`, set `lambda_r = 1` and all other weights to zero, and `g^te(x* | lambda, z*) = 0`, the global minimum because no feasible point can go below `z*_r`. If I insist on strictly positive weights, I get the same case as a limit, with the exact Pareto-optimality guarantee weakened in the usual way. The converse also needs the right strength: with strictly positive weights, a minimizer cannot be strictly dominated in every objective, because that would lower every weighted gap and hence lower the max, so every minimizer is at least weakly Pareto-optimal. If the minimizer is unique, even a point that is no worse in all objectives and better in one would have `g^te(y) <= g^te(x)`, contradicting uniqueness; a unique minimizer is Pareto-optimal. Every front point, convex or not, is reachable in the sense I need. The price is that `g^te` is non-smooth — that `max` has a kink — but I'm using a derivative-free EA, so I never differentiate it; the kink costs me nothing. Tchebycheff it is, as the default.

I should keep one more scalarization available, because I can already foresee that with three or more objectives even Tchebycheff might not spread points as evenly as I'd like. The boundary-intersection idea (Das & Dennis's normal-boundary-intersection, Messac's normalized normal constraint) shoots a line from the reference point in direction `lambda` and finds where it pierces the front; evenly spread lines give evenly spread piercing points, even on nonconvex fronts. The clean form has an awkward equality constraint `F(x) - z* = d*lambda`, so I'd handle it with a penalty. Let `u = lambda / ||lambda||`. Then `d_1 = |(F(x) - z*)^T u|` is the distance of `F(x)` along the line through `z*`, and `d_2 = ||(F(x) - z*) - d_1 u|| = ||F(x) - (z* + d_1 u)||` is the perpendicular distance off that line. The scalar objective becomes `g^pbi(x | lambda, z*) = d_1 + theta*d_2`, with `theta > 0` penalizing off-line drift. Geometrically I'm pulling `F(x)` to the first feasible boundary point along the prescribed line while punishing sideways movement away from the line. The reason to bother: with the *same* set of evenly spread weights, the line-projection geometry forces solutions onto the prescribed directions, so for `m >= 3` it lays them out more uniformly than the corner-contours of Tchebycheff do. The cost is a knob — `theta` too large or too small both degrade it. So I'll default to Tchebycheff for two objectives and reach for PBI when there are three or more and uniformity matters; the framework is identical either way, only the aggregation function `g` swaps out.

Now, scalarization gives me a recipe — *pick a weight, optimize the scalar problem* — but how do I turn the recipe into one population-based run that produces the whole spread? There's already a method that took the scalarization route, MOGLS by Jaszkiewicz (out of Ishibuchi and Murata). Its move each iteration: draw a weight vector `lambda` *at random*, gather the `K` best current solutions under `g^te(. | lambda)` into a temporary elite pool, recombine and locally-improve two of them, and drop the result into a big current set `CS` (which it lets balloon to thousands), with an external archive of non-dominated points on the side. So it *is* solving scalar subproblems — but watch the inefficiency. The weight is fresh and random every single iteration, so the effort smears across effectively infinitely many subproblems. But the decision maker only wanted a finite, evenly spread handful of representatives! Pouring optimization into a continuum of subproblems, most of which I'll never report, is waste. And forming that temporary elite pool means scanning `CS` to find the `K` best for the current `lambda`, which is `O(K |CS|)` with `|CS|` in the thousands — every iteration. Two things are clearly wrong here: the subproblems should be a *fixed, finite* set chosen up front to be exactly the representatives I want, and I shouldn't be paying to re-rank a giant pool against a random weight each step. Wall, but an instructive one.

So let me commit to the fixed finite set. Choose `N` weight vectors once, spread evenly over the simplex, and let each one *be* a subproblem I will optimize and ultimately report. How do I spread `N` weights evenly? Das and Dennis's simplex-lattice: take every vector whose entries come from `{0/H, 1/H, ..., H/H}` and sum to one. That places points on a regular grid over the unit simplex, and the count is `N = C(H + m - 1, m - 1)`, fixed by the number of objectives `m` and the granularity `H`. For two objectives this is just `lambda = (i/(N-1), 1 - i/(N-1))` for `i = 0, ..., N-1`, an even fan of directions. Good — `N` subproblems, each pinned to one weight, each will hold one current best solution. The population *is* the roster of these `N` current bests, one per subproblem. Already this is different in kind from NSGA-II: there's no global pool to non-dominated-sort; each population slot has an *owner* (its subproblem) and a total order (its scalar `g`), so selection for that slot never goes silent.

If the `N` weights are evenly spread and each subproblem is well-optimized, then each subproblem targets a different trade-off direction along the front — so the *population itself*, being one solution per evenly-spaced subproblem, is organized across the front by construction. I don't need crowding distance or a density grid or a `k`-NN estimator to create spread pressure; the distribution of weights *is* the diversity mechanism. The thing NSGA-II had to manufacture with a separate device, decomposition produces as a side effect of the structure. That's the payoff I was chasing — total order for pressure, weight directions for spread, both intrinsic.

But now I have `N` separate scalar optimizations and a tiny budget — one solution per subproblem. If I optimize each subproblem in total isolation, each is just a single-point hill-climb with no population to recombine within, and I've thrown away the whole reason to use an EA. I need the subproblems to *help each other*. What's the structure I can exploit? Stare at two weight vectors that are close to each other, `lambda^i ≈ lambda^j`. Their Tchebycheff functions `g^te(. | lambda^i, z*)` and `g^te(. | lambda^j, z*)` are nearly the same function — `g^te` depends continuously on `lambda`, so a small change in the weight is a small change in the scalar objective, and therefore the optimizer of subproblem `i` is close to the optimizer of subproblem `j`. That's the key fact, and it's worth saying precisely why it matters: the current best solution to subproblem `j` is, for free, a *good* candidate solution to subproblem `i`, because their optima nearly coincide. So neighboring subproblems are exactly the solutions I should be recombining and sharing information among.

That suggests defining, for each subproblem `i`, a neighborhood `B(i)` = the `T` subproblems whose weight vectors are closest to `lambda^i` (Euclidean distance in weight space; `i` is its own nearest neighbor, so `i in B(i)`). Then I run the loop: to make progress on subproblem `i`, pick two of its neighbors at random as parents, breed an offspring, and try it not just on subproblem `i` but on *every* neighbor `j in B(i)`, replacing the current solution of subproblem `j` whenever the offspring scores better on `g^te(. | lambda^j, z*)`. A single good child therefore propagates outward to every nearby subproblem it improves, so information flows along the chain of weights. This is what replaces NSGA-II's `O(m N^2)` non-dominated sort: there is no global sort at all. Each of the `N` passes does `O(1)` to pick parents, `O(m)` to update the reference point, and `O(mT)` to test the offspring against the `T` neighbors — so a generation is `O(mNT)`. Against NSGA-II's `O(m N^2)`, the ratio is `O(T)/O(N)`, smaller because the neighborhood `T` is much smaller than the population `N`. Cheaper *and* the diversity is free. The pieces have clicked together.

Let me size `T`, because it's the one real knob in the structure and the failure modes are opposite at the two extremes. If `T` is too small, the parents drawn from `B(i)` are nearly identical solutions to near-identical subproblems, so the offspring barely differs from them and exploration stalls — and a child that only ever gets tried on a couple of subproblems can't spread. If `T` is too large, the neighbors are solutions to *distant* subproblems with very different optima, so they make poor parents for subproblem `i` (recombining solutions from opposite ends of the front rarely produces something good for the middle), exploitation weakens, and the `O(mT)` neighbor-update cost climbs. So `T` wants to be a moderate fraction of `N` — large enough to give recombination some variety, small enough that the neighbors are genuinely "near." A value around `T = 20` is the standard choice and sits comfortably in that band for the population sizes I'm using.

I need the ideal point `z*` for Tchebycheff, but `z*_i = min_{x in Omega} f_i(x)` would mean solving `m` single-objective problems exactly — defeating the purpose. So I substitute `z`, the running minimum of each objective over every solution I've evaluated, updated whenever a new solution beats the current `z_j` on objective `j`. Tchebycheff only really needs `z` to sit at or below the front (so the corner-contours open the right way); the running best converges to that from above as the search progresses, and it's free. Initialize `z` from the first evaluated population. There's a subtlety I should keep in mind: because `z` drifts as new bests appear, a `g^te` value computed before a drift isn't directly comparable to one computed after it — but I always compare offspring vs incumbent *at the same `z`* within a single neighbor-update step, so the drift doesn't corrupt any individual comparison.

Now I'll be concrete about the per-generation procedure, the Tchebycheff version. Initialize: compute the `N` weight vectors; for each `i`, compute `B(i)` as the `T` nearest weights; generate `N` random initial solutions `x^1, ..., x^N`, one per subproblem; evaluate them; set `z` to the per-objective minimum. Then each generation, for `i = 1, ..., N`: pick two indices `k, l` at random from `B(i)`; produce an offspring `y` from `x^k, x^l` by the genetic operators; update `z` — for each objective `j`, if `f_j(y) < z_j` set `z_j = f_j(y)`; then for each `j in B(i)`, if `g^te(y | lambda^j, z) < g^te(x^j | lambda^j, z)` set `x^j = y`. If I keep an external archive, I update it with the non-dominated objective vectors seen so far; for a head-to-head against NSGA-II, which has no archive, I can instead return the final `N` internal solutions, drop the archive, and the only extra memory over NSGA-II is the `O(m)` vector `z`. That's it — no front sorting, no crowding, no density estimate anywhere.

The strict loop draws parents from `B(i)`. In code I can expose that as a probability `delta`: `delta = 1` recovers the strict neighborhood rule exactly, while `delta < 1` keeps the usual local mating most of the time and occasionally draws from the whole population to inject material from far away when local fronts or multimodal landscapes make a neighborhood too greedy. I still try the offspring only against the source neighborhood, so one lucky child cannot overwrite an unbounded number of slots and collapse diversity. A value `delta = 0.9` keeps the search mostly local with a 10% chance of a global jump.

For the offspring themselves I need the variation operators, and since the decision space is a real box `[x_l, x_u]^n` I use the standard real-coded pair from the NSGA-II lineage, both for fairness (so any difference is the *strategy*, not the operators) and because they're well understood. Simulated binary crossover with distribution index `eta_c`: it samples a spread factor `beta_q` whose distribution is polynomial in `beta` with exponent set by `eta_c`, then places the two children symmetrically about the parents at spread `beta_q`; a large `eta_c` concentrates children near the parents, a small one spreads them. The bounded version computes `beta` from the parent positions relative to each box edge so the children stay inside `[x_l, x_u]`. Polynomial mutation with distribution index `eta_m`: each variable, with probability `p_m`, gets perturbed by `delta_q` drawn from a polynomial distribution (again box-respecting), large `eta_m` meaning small perturbations. The field-standard defaults are `eta_c = eta_m = 20` (moderate spread), crossover probability `1.0` (always recombine), and `p_m = 1/n` so on average about one variable mutates per child — enough to keep exploring without destroying good solutions. I make the implementation produce *one* child per mating (use the first SBX offspring), since each subproblem only needs a single trial solution per pass.

For three or more objectives I swap the aggregation: build the weights with the same simplex-lattice (now genuinely `N = C(H + m - 1, m - 1)` points on a 2-simplex for `m = 3`) and use PBI `g^pbi = d_1 + theta*d_2` instead of `g^te` in the neighbor-update, because, as I argued, the line-projection geometry spreads three-objective fronts more uniformly than the Tchebycheff corners. Everything else in the loop is unchanged.

Let me also pin the complexity claim down, since it's half the reason this is worth doing. NSGA-II's cost is dominated by the fast non-dominated sort over `2N` points, `O(m N^2)` per generation. My generation does `N` passes; each pass is `O(1)` to pick parents from `B(i)`, `O(m)` to update `z`, and `O(mT)` to evaluate the offspring against the `T` neighbors' aggregation values — `O(mT)` per pass, `O(mNT)` per generation. So the ratio of per-generation costs is `O(mNT)/O(m N^2) = O(T)/O(N)`, and when `T << N` the neighborhood update is asymptotically cheaper than the global sort. Memory is one solution per subproblem plus the `O(m)` reference point — linear in the population.

Now let me write it as code I'd actually run, against a generic multi-objective EA harness whose three empty hooks are `select` (who mates), `vary` (produce offspring), and `survive` (who carries forward). Decomposition doesn't fit the global parent-pool-then-sort pattern, so I let `select` just hand back the current roster, do the neighborhood mating inside `vary` (pairing each subproblem's offspring with neighbor parents), and put the reference-point update plus decomposition-based replacement in `survive`. The weights, neighborhoods, and reference point are state set up once. The clean numpy structure is direct: neighbors via `argsort(cdist(weights, weights))[:, :T]` (which includes each subproblem itself), the ideal as the per-objective min, Tchebycheff by `max_j lambda_j |f_j - z_j|` for two objectives, PBI by `d_1 + theta d_2` for three or more, and neighbor replacement whenever the offspring's score is lower.

```python
import random
from copy import deepcopy
from math import comb

import numpy as np
from scipy.spatial.distance import cdist
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """MOEA/D: decompose the MOP into N scalar subproblems, one per evenly
    spread weight vector; each subproblem keeps one current solution; mate and
    share information only among neighboring (weight-adjacent) subproblems."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index (default 20)
        self.mut_eta = mut_eta                     # polynomial-mutation index (default 20)
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.delta = 0.9                           # prob. of mating within the neighborhood
        self.theta = 5.0                           # PBI penalty

        # N evenly spread weight vectors on the unit simplex (Das-Dennis lattice).
        self.H = self._lattice_resolution(pop_size, n_obj)
        self.weights = self._generate_weights(pop_size, n_obj, self.H)
        self.pop_size = len(self.weights)          # actual N from the lattice count
        self.T = min(20, self.pop_size)            # neighborhood size

        # B(i): the T weight-nearest subproblems (argsort over pairwise weight
        # distances; column 0 is i itself, so each B(i) contains i).
        dist = cdist(self.weights, self.weights)
        self.neighbors = [np.argsort(dist[i])[:self.T].tolist()
                          for i in range(self.pop_size)]

        self.z_star = None                         # running ideal point z (per-objective min)
        self._offspring_sources = []

    def _lattice_resolution(self, n, n_obj):
        if n_obj == 2:
            return max(n - 1, 1)                   # C(H+1,1) = H+1 = N
        H = 1
        while comb(H + n_obj - 1, n_obj - 1) < n:  # N = C(H+m-1,m-1)
            H += 1
        return H

    def _generate_weights(self, n, n_obj, H):
        if n_obj == 2:                             # even fan of bi-objective directions
            if n <= 1:
                return np.array([[0.5, 0.5]])
            return np.array([[i / max(n - 1, 1), 1.0 - i / max(n - 1, 1)]
                             for i in range(n)])
        # m >= 3: simplex-lattice points, N = C(H+m-1, m-1).
        return np.array(tools.uniform_reference_points(n_obj, p=H), dtype=float)

    def _tchebycheff(self, fvals, weight, z):
        # g^te = max_j lambda_j * |f_j - z_j|
        return max(weight[j] * abs(fvals[j] - z[j]) for j in range(self.n_obj))

    def _pbi(self, fvals, weight, z):
        # PBI distance: project onto the normalized weight direction.
        diff = np.asarray(fvals, dtype=float) - np.asarray(z, dtype=float)
        w = np.asarray(weight, dtype=float)
        norm = np.linalg.norm(w)
        u = w / norm
        d1 = float(abs(np.dot(diff, u)))
        d2 = float(np.linalg.norm(diff - d1 * u))
        return d1 + self.theta * d2

    def _decompose(self, fvals, weight, z):
        if self.n_obj <= 2:
            return self._tchebycheff(fvals, weight, z)
        return self._pbi(fvals, weight, z)

    def select(self, population, k):
        # Decomposition uses no global parent selection; the roster IS the
        # population, one solution per subproblem. Hand it back for mating.
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # One offspring per subproblem i, bred from two parents drawn (mostly)
        # from B(i): close subproblems have close optima, so neighbors are good
        # mates; with prob 1-delta mate globally to escape local fronts.
        offspring = []
        lo, hi = self.bounds
        order = random.sample(range(len(parents)), len(parents))
        self._offspring_sources = order
        for i in order:
            if random.random() < self.delta:
                pool = [parents[j] for j in self.neighbors[i]]
            else:
                pool = parents
            if len(pool) < 2:
                pool = parents
            a, b = random.sample(range(len(pool)), 2)
            child = deepcopy(pool[a])
            mate = deepcopy(pool[b])
            tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)  # p_c = 1
            tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi,
                                       indpb=self.mut_prob)
            del child.fitness.values               # invalidate so the harness re-evaluates
            offspring.append(child)
        return offspring

    def survive(self, population, offspring):
        # The heart: initialize z from evaluated parents, then for each evaluated
        # offspring update z and replace the neighboring subproblems it beats.
        next_pop = list(population)
        if not next_pop:
            return population
        if self.z_star is None:
            self.z_star = [float('inf')] * self.n_obj
            for ind in next_pop:
                if ind.fitness.valid:
                    for j in range(self.n_obj):
                        if ind.fitness.values[j] < self.z_star[j]:
                            self.z_star[j] = ind.fitness.values[j]

        sources = self._offspring_sources or list(range(len(offspring)))
        for source_i, child in zip(sources, offspring):
            if not child.fitness.valid:
                continue
            for j in range(self.n_obj):             # z_j <- min over all seen f_j
                if child.fitness.values[j] < self.z_star[j]:
                    self.z_star[j] = child.fitness.values[j]
            for j in self.neighbors[source_i]:      # try child on every neighbor j of source_i
                if j >= len(next_pop):
                    continue
                g_child = self._decompose(child.fitness.values, self.weights[j], self.z_star)
                g_cur = self._decompose(next_pop[j].fitness.values, self.weights[j], self.z_star)
                if g_child < g_cur:                 # offspring better for subproblem j -> take over
                    next_pop[j] = deepcopy(child)
        return next_pop[:self.pop_size]

    def on_generation(self, gen, population):
        pass
```

So the causal chain is this. I started stuck with dominance, which gives only a partial order, so once the population is mutually non-dominated the pressure toward the front goes silent and diversity has to be supplied by a separate, costly, bolt-on device (crowding distance, density grids), with an `O(m N^2)` sort each generation and no way to reuse the rich classical scalarization theory. Scalarization promised a total order per subproblem, so I tried weighted-sum — but its hyperplane contours can't reach into nonconvex dents of the front, missing whole regions. Tchebycheff's corner contours fix that: by aligning `lambda` with the ray from the ideal point I showed every Pareto point can be made an optimizer of some `g^te`, convex or not, and the non-smoothness is free for a derivative-free EA. MOGLS had taken the scalarization route but wasted effort on a fresh random weight every iteration and paid `O(K|CS|)` per step over a giant pool, so I fixed a finite, evenly spread set of `N` weights (Das-Dennis simplex-lattice), made each one a subproblem holding one current solution, and noticed that the weight directions make population spread an intrinsic pressure rather than a separate density calculation. To let the `N` thin single-solution subproblems help each other I exploited that weight-adjacent subproblems have adjacent optima, defining each subproblem's neighborhood as its `T` nearest weights, mating within it, and letting each offspring update every neighbor it improves — which removes the global sort entirely and drops the per-generation cost to `O(mNT)`, a factor `O(T)/O(N)` cheaper than NSGA-II. I sized `T` (moderate: too small kills exploration, too large kills exploitation and costs more), substituted a running ideal `z` for the unknown `z*`, added a `delta`-probability global mating to escape local fronts, carried the standard SBX/polynomial-mutation operators with the field defaults, and swapped in PBI for three-or-more objectives where uniformity needs the line-projection geometry. The result is one population-based run that keeps per-subproblem selection pressure, organizes spread through weight directions instead of a separate diversity device, and costs less per generation than the dominance-based state of the art.
