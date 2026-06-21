The task is to approximate the Pareto front of a multi-objective problem $\min F(x) = (f_1(x), \dots, f_m(x))$ over a feasible set $\Omega$, where the objectives conflict so that no single $x$ minimizes all of them. What a decision maker actually wants is not the whole, often infinite, front but a finite, evenly spread, well-converged handful of representative trade-offs computed in one run — points that are simultaneously close to the true front (convergence) and uniformly distributed along it (diversity), for fronts that may be convex, concave, disconnected, or, with three or more objectives, curved surfaces. The dominant tool of the day is dominance-based evolutionary search, and dominance is exactly where the difficulty lives. Dominance is only a *partial* order: if neither of two candidates dominates the other, it cannot say which is better. Early in a run that is harmless, since many points are plainly bad and get dominated away; but as the population improves, more and more of it becomes mutually non-dominated, and then dominance falls silent — it can no longer separate members of the first front. NSGA-II feels this directly: it pools parents and offspring, runs an $O(m N^2)$ fast non-dominated sort into nested fronts, and fills the next generation front by front, breaking the last tie with *crowding distance*, $\sum_i (f_i^{\text{next}} - f_i^{\text{prev}})/(f_i^{\max} - f_i^{\min})$. Two things are wrong with this. Once everyone sits in the first front, the only thing deciding survival is that crowding tie-break — a separate density heuristic bolted on to manufacture the diversity that selection pressure was supposed to provide. And with three or more objectives almost every pair is mutually non-dominated, so dominance carries essentially no information while the sort stays just as expensive. SPEA2 and PAES are the same family: rank by dominance, then staple on a density device (strength plus $k$-NN, or an adaptive grid). All of them treat the MOP as one indivisible blob, never tying any individual to a scalar problem, and so they throw away the entire mature body of classical scalarization theory.

I propose MOEA/D — a Multiobjective Evolutionary Algorithm based on Decomposition. The move is to stop optimizing the vector problem as a whole and instead *decompose* it into $N$ scalar subproblems, one per weight vector, and optimize them simultaneously and cooperatively in a single population-based run. A weight vector $\lambda = (\lambda_1, \dots, \lambda_m)$, $\lambda_i \geq 0$, $\sum_i \lambda_i = 1$, fixes a trade-off preference, and a scalar aggregation function $g(x \mid \lambda, z^*)$ turns that preference into a single objective with a *total* order — for any two solutions I can always say which is better for that subproblem. That total order is precisely the sustained selection pressure dominance lacks. The first scalarization one reaches for is the weighted sum $g^{\text{ws}}(x \mid \lambda) = \sum_i \lambda_i f_i(x)$, but it fails the one test that matters: its level sets are hyperplanes with normal $\lambda$, and minimizing slides that hyperplane until it last touches the attainable set, so it can only reach *supported* Pareto points on the boundary of the convex hull. A hyperplane sliding into a re-entrant, nonconvex stretch of the front skips over the dent and touches only the convex shoulders, so points in nonconvex or disconnected regions — exactly what ZDT2, ZDT3 and friends are made of — are unreachable for any $\lambda$. The fix is to change the contour geometry, which is what the Tchebycheff scalarization does:
$$g^{\text{te}}(x \mid \lambda, z^*) = \max_i \{\, \lambda_i\, |f_i(x) - z^*_i| \,\},$$
where $z^* = (z^*_1, \dots, z^*_m)$ is the ideal point, $z^*_i = \min_{x \in \Omega} f_i(x)$. Its level sets are nested axis-aligned right-angle corners (rescaled $L_\infty$ balls) with their vertex marching out from $z^*$ along the ray $1/\lambda$; a corner can reach *into* a dent where no flat hyperplane can. The optimality argument is what makes this load-bearing: take a Pareto-optimal $x^*$ and write the gaps $d_i = f_i(x^*) - z^*_i$. If every $d_i > 0$, choose $\lambda_i \propto 1/d_i$, so after normalization every product $\lambda_i d_i$ equals one constant $c$, giving $g^{\text{te}}(x^* \mid \lambda, z^*) = c$. If some feasible $y$ scored below $c$, then $\lambda_i (f_i(y) - z^*_i) < c = \lambda_i (f_i(x^*) - z^*_i)$ for every $i$, hence $f_i(y) < f_i(x^*)$ for all $i$ — $y$ would dominate $x^*$, which is impossible. So $x^*$ is a minimizer; zero-gap coordinates are handled as the limiting weak-weight case. With strictly positive weights every minimizer is at least weakly Pareto-optimal, and a unique minimizer is Pareto-optimal, so sweeping $\lambda$ traces the *whole* front, convex or not. The only blemish is the kink in the $\max$, which costs nothing to a derivative-free EA that never differentiates $g^{\text{te}}$. So Tchebycheff is the default. I keep one more aggregation in reserve for $m \geq 3$, the penalty boundary intersection: with $u = \lambda/\|\lambda\|$, $d_1 = |(F(x) - z^*)^\top u|$ the distance along the line through $z^*$ in direction $\lambda$ and $d_2 = \|(F(x) - z^*) - d_1 u\|$ the perpendicular drift off it,
$$g^{\text{pbi}}(x \mid \lambda, z^*) = d_1 + \theta\, d_2, \qquad \theta > 0,$$
which pulls $F(x)$ to the first boundary point along the prescribed line while punishing sideways movement. With the *same* evenly spread weights, this line-projection geometry lays many-objective fronts out more uniformly than Tchebycheff's corners, at the cost of tuning $\theta$ (a value of $5$ works well; too large or too small both degrade it).

The recipe "pick a weight, optimize the scalar problem" still has to become one population-based run. MOGLS took the scalarization route but drew a *fresh random* weight every iteration, smearing effort across effectively infinitely many subproblems — most never reported — and paid $O(K|CS|)$ each step re-ranking a pool that balloons to thousands. The fix is to commit to a *fixed, finite* set of subproblems that are exactly the representatives I want: choose $N$ weight vectors once, evenly spread on the simplex by the Das–Dennis lattice (every vector with entries from $\{0/H, 1/H, \dots, H/H\}$ summing to one, count $N = \binom{H + m - 1}{m - 1}$; for $m = 2$ the even fan $\lambda = (i/(N-1),\, 1 - i/(N-1))$). Each weight *is* a subproblem that holds one current solution, so the population is the roster of these $N$ bests — one per evenly spaced subproblem. This already buys two things at once. Every population slot has an owner and a total scalar order, so its selection never goes silent. And because the weights are evenly spread, each subproblem targets a different trade-off direction, so the population is organized across the front *by construction* — the distribution of weights *is* the diversity mechanism, replacing crowding distance, density grids, and $k$-NN with nothing at all. The thing NSGA-II had to manufacture, decomposition gets as a side effect of its structure.

What remains is that $N$ isolated single-solution hill-climbs would throw away the reason to use an EA, so the subproblems must help each other. The structure to exploit is that if two weights are close, $\lambda^i \approx \lambda^j$, then $g^{\text{te}}(\cdot \mid \lambda^i, z^*)$ and $g^{\text{te}}(\cdot \mid \lambda^j, z^*)$ are nearly the same function (the aggregation depends continuously on $\lambda$), so their optima nearly coincide — the current best of subproblem $j$ is, for free, a *good* candidate for subproblem $i$. So I define each subproblem's neighborhood $B(i)$ as the $T$ subproblems with the weight-nearest vectors (Euclidean in weight space; $i \in B(i)$, since each weight is its own nearest neighbor), and run the loop: to advance subproblem $i$, pick two parents at random from $B(i)$, breed one offspring $y$, update the ideal point $z$ coordinate-wise by $z_j \leftarrow \min(z_j, f_j(y))$, and then try $y$ against *every* neighbor $j \in B(i)$, replacing $x^j$ with $y$ whenever $g^{\text{te}}(y \mid \lambda^j, z) < g^{\text{te}}(x^j \mid \lambda^j, z)$. A single good child therefore propagates outward to every nearby subproblem it improves, so information flows along the chain of weights, and there is no global sort anywhere. Each of the $N$ passes is $O(1)$ to pick parents, $O(m)$ to update $z$, and $O(mT)$ to test the offspring against the $T$ neighbors, so a generation costs $O(mNT)$ against NSGA-II's $O(m N^2)$ — a ratio of $O(T)/O(N)$, cheaper exactly because $T \ll N$. The neighborhood size $T$ is the one real knob, with opposite failure modes: too small and the parents from $B(i)$ are near-identical, so exploration stalls and a child reaches too few subproblems; too large and the mates come from distant subproblems with very different optima, so recombination produces poor children and the $O(mT)$ update cost climbs. A moderate $T \approx 20$ sits in the safe band. Since $z^*_i = \min_{x\in\Omega} f_i(x)$ would mean solving $m$ single-objective problems, I substitute the running per-objective minimum $z$, which Tchebycheff only needs to sit at or below the front; offspring and incumbent are always compared at the *same* $z$ within one update step, so the drift of $z$ never corrupts a comparison. To escape local fronts I expose the neighbor-mating restriction as a probability $\delta$: with probability $\delta = 0.9$ I draw parents from $B(i)$, and with probability $1 - \delta$ from the whole population, injecting far-away material — but I still try the child only against the source neighborhood, so one lucky offspring cannot overwrite an unbounded number of slots and collapse diversity. The offspring use the standard real-coded NSGA-II pair so any measured difference is the *strategy*, not the operators: simulated binary crossover with distribution index $\eta_c = 20$ at probability $1.0$, and polynomial mutation with $\eta_m = 20$ and per-variable rate $p_m = 1/n$, taking one child per mating since each subproblem needs only one trial per pass. For $m \geq 3$ everything is unchanged except that $g^{\text{pbi}}$ replaces $g^{\text{te}}$ in the neighbor update.

```python
import random
from copy import deepcopy
from math import comb

import numpy as np
from scipy.spatial.distance import cdist
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """MOEA/D: Multi-Objective Evolutionary Algorithm based on Decomposition.
    Decomposes the MOP into N scalar subproblems on evenly spread weight
    vectors; each subproblem keeps one current solution and cooperates only
    with its T weight-nearest neighbors."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index
        self.mut_eta = mut_eta                     # polynomial-mutation distribution index
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
        # No global parent selection: the roster IS the population, one solution
        # per subproblem. Hand it back for neighborhood mating in vary().
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # One offspring per subproblem i, from two parents drawn (with prob delta)
        # from B(i) -- close subproblems have close optima, so neighbors mate well;
        # with prob 1-delta mate globally to escape local fronts.
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
