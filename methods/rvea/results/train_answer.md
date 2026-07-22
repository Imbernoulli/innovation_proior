We want a single evolutionary run that returns a finite, evenly spread, well-converged sample of the Pareto front of a many-objective problem $\min F(x) = (f_1(x), \dots, f_M(x))$ with $M > 3$. The honest answer to such a problem is never one point but the non-dominated set, and with a fixed population we can only return a sample of it, so the population has to do two things at once: drive toward the front (convergence) and stay evenly distributed across it (diversity). For two or three objectives the standard machine — Pareto dominance to converge, a density device to keep the sample spread, exactly NSGA-II — works beautifully. The reason it falls apart as $M$ grows is worth being precise about, because the fix follows from the failure. Pick two candidate solutions in a high-dimensional objective space; for one to dominate the other it must be no worse on all $M$ objectives, and as $M$ rises the chance that one vector beats another on every coordinate plummets, so almost every pair comes out mutually non-dominated. The first non-dominated front swallows nearly the whole pool within a handful of generations, the dominance sort goes silent because it puts everyone in the same front, and the only thing left deciding survival is the crowding-distance tie-break — a density heuristic designed to break ties, not to converge, and itself degraded because the population sits sparsely in high dimensions and its per-objective neighbour gaps become noise. Both halves of NSGA-II break, and they break precisely because dominance is a partial order that goes blind exactly when everything is incomparable. Scalarization-based methods point at a remedy — MOEA/D attaches a scalar sub-problem to each of $N$ spread weight vectors, and its PBI aggregation $g^{\text{pbi}} = d_1 + \theta\,d_2$ fuses a convergence term $d_1$ (displacement along the line) and a diversity term $\theta\,d_2$ (Euclidean perpendicular distance off it) into one total order per direction. That skeleton is right, but its diversity term is wrong twice over: $\theta$ is a single fixed scalar with no setting that works across problems and objective counts, and $d_2$ is an unbounded Euclidean distance that grows simply with how far a solution sits from the ideal point, so it is entangled with convergence rather than a clean diversity reading. NSGA-III's remedy for badly-scaled fronts — normalizing the objectives onto the unit hyperplane every generation — is safe for a dominance method because monotone per-objective rescaling preserves the partial order, but it mangles any metric criterion: it would change the very numbers a geometric criterion reads, and change them differently every generation.

I propose RVEA, the Reference Vector Guided Evolutionary Algorithm. The core is to replace dominance entirely with a total order built from two genuinely orthogonal readings of each solution, and to attach one such criterion to each of $N$ reference directions, selecting one survivor per direction. I generate the directions with the Das-Dennis simplex-lattice — every vector $u_i$ with entries in $\{0/H, \dots, H/H\}$ summing to one, giving $N = \binom{H+M-1}{M-1}$ evenly placed points — but because my criterion is angular I map each off the simplex onto the unit hypersphere, $v_i = u_i / \lVert u_i \rVert$, so these are directions, not weights, and any cosine taken against them needs no renormalization on their side. The two readings come from translating the objectives. Using the running per-objective minimum $z^{\min}$ over the evaluated pool as the cheap substitute for the unknown ideal point, I set $f' = f - z^{\min}$, which pins the ideal to the origin and puts every $f'$ in the first quadrant; then $\lVert f' \rVert$ *is* the distance to the ideal — the convergence reading — for free, and the acute angle of $f'$ to a reference vector is the diversity reading. The angle is exactly what PBI's $d_2$ should have been: it is bounded in $[0, \pi/2]$ and scale-invariant, since $\cos\theta = (a \cdot b)/(\lVert a \rVert\,\lVert b \rVert)$ is homogeneous of degree zero, so however far out a solution lies its angle to a reference vector does not move. Length for convergence, angle for diversity, decoupled — instead of PBI's two entangled distances.

Selection then runs on the combined parent-plus-offspring pool of the $\mu+\lambda$ shell. I associate each individual with the reference vector of minimum acute angle (equivalently maximum cosine, since $v_j$ is unit), which partitions the pool into $N$ subpopulations, and from each non-empty subpopulation I keep the single best individual by a scalar I call the angle-penalized distance, the APD. Crucially I *scale* the convergence distance by the diversity penalty rather than *adding* it, because addition is what entangled PBI's terms. Writing $\theta_{i,j}$ for the angle between $f'_i$ and its associated vector $v_j$,
$$d_{i,j} = \bigl(1 + P(\theta_{i,j})\bigr)\,\lVert f'_i \rVert, \qquad P(\theta_{i,j}) = M \,\Bigl(\tfrac{t}{t_{\max}}\Bigr)^{\alpha}\,\frac{\theta_{i,j}}{\gamma_{v_j}}.$$
A solution perfectly on-direction has $\theta = 0$, so $P = 0$ and $d = \lVert f' \rVert$ — among on-direction solutions I simply keep the one closest to the front — while an off-direction solution has its distance inflated in proportion to how far out it already sits, which is right, since poaching a niche matters more the further from the front it happens. Each factor of $P$ is derived from the search dynamics rather than guessed. The ramp $(t/t_{\max})^\alpha$ encodes that spreading a population that has not yet converged just spreads it in the wrong place: early in the run I want near-pure convergence, late I want spread, so the penalty should start near zero and grow, and raising the generation ratio to $\alpha = 2$ gives a convex ramp that stays nearly flat through the early and middle game then climbs steeply near $t_{\max}$. The division by $\gamma_{v_j} = \min_{i \neq j} \angle(v_i, v_j)$, each vector's nearest-neighbour angle, measures $\theta$ relative to the niche's own width: a deviation that already overlaps a close neighbour's territory should be penalized, the same deviation inside a wide niche should not, so normalizing makes the penalty comparable across dense and sparse regions of the reference set — and it normalizes the angle, not the objectives, leaving the convergence length untouched. The factor $M$ corrects for objective-space sparsity growing with the number of objectives. Within each subpopulation the survivor is $\arg\min_i d_{i,j}$, and the roster of one survivor per non-empty vector is the next generation; no dominance comparison appears anywhere, so selection is a total order per subspace and can never go blind.

Badly-scaled fronts, where objectives differ by orders of magnitude, I cannot fix by normalizing the objectives, because that would corrupt the very angle-and-length geometry the criterion reads — translated vectors $(0.1, 2)$ and $(1, 10)$ are separated by $\lVert(0.9, 8)\rVert \approx 8.05$, but per-objective range normalization sends them to $(0.1, 0.2)$ and $(1, 1)$, collapsing the separation to $\lVert(0.9, 0.8)\rVert \approx 1.2$, and doing so afresh every generation as the scales drift. So I move the reference vectors instead of the objectives. Every $\lceil f_r\,t_{\max} \rceil$ generations I estimate the per-objective range $z^{\max} - z^{\min}$ of the current population, stretch each *pristine* vector by it, and renormalize,
$$v_{t+1,i} = \frac{v_{0,i} \circ (z^{\max} - z^{\min})}{\lVert v_{0,i} \circ (z^{\max} - z^{\min}) \rVert},$$
with $\circ$ the Hadamard product, always rebuilding from the original $v_0$ so scaling errors do not accumulate, and the objectives stay exactly where they are so the criterion remains coherent. The adaptation is occasional, not per-generation: if the vectors moved every generation the niches would shift under the population and convergence would chase a moving target, so with $f_r = 0.1$ there are about ten corrections across the run — enough to track a drifting scale, rare enough to stay stable — and because the vectors move, $\gamma$ is recomputed each time. The offspring side is deliberately plain: every survivor is already the elitist of its own subspace, so there is no reason to bias mating, and I sample parents uniformly, pair them at random, and apply SBX with $\eta_c = 30$ (a touch more local than the usual $20$, so well-placed niche survivors are not disrupted) and polynomial mutation with $\eta_m = 20$, $p_c = 1.0$, $p_m = 1/n$. The whole generation is $O(MN^2)$, dominated by the angular partition — the same order as NSGA-II's non-dominated sort, but with no dominance comparison and none of its partial-order pathology.

```python
import random
from copy import deepcopy

import numpy as np
from deap import tools   # cxSimulatedBinaryBounded, mutPolynomialBounded, uniform_reference_points


class CustomMOEA:
    """RVEA: Reference Vector Guided Evolutionary Algorithm. Selects one survivor
    per reference vector by the angle-penalized distance (a scaled distance-to-ideal,
    inflated by a dynamic, niche-normalized angle penalty); adapts the reference
    vectors to the objective scales rather than normalizing the objectives."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=30.0, mut_eta=20.0, mut_prob=None):
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds                       # (low, up) of the box decision space
        self.cx_eta = cx_eta                       # SBX distribution index (RVEA uses 30)
        self.mut_eta = mut_eta                     # polynomial-mutation index (20)
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.alpha = 2.0                           # APD penalty ramp exponent
        self.fr = 0.1                              # reference-vector adaptation frequency
        self.max_gen = 400                         # horizon for the (t/t_max)^alpha ramp;
                                                   #   the driver can overwrite it per problem

        # N evenly spread reference points (Das-Dennis), mapped to the unit sphere.
        p = pop_size - 1 if n_obj == 2 else 12
        V0 = np.asarray(tools.uniform_reference_points(n_obj, p=p), dtype=float)
        self.V0 = self._unit(V0)                   # pristine unit reference vectors
        self.V = self.V0.copy()                    # current (adaptable) reference vectors
        self.pop_size = len(self.V0)
        self.gamma = self._gamma(self.V)           # per-vector nearest-neighbour angle
        self.z_min = None                          # running ideal (per-objective min)
        self._gen = 0

    @staticmethod
    def _unit(V):
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        return V / norms

    def _gamma(self, V):
        # smallest acute angle from each v_j to any OTHER reference vector:
        # arccos of the second-largest cosine in each row (largest = v_j with itself).
        cos = np.clip(V @ V.T, -1.0, 1.0)
        cos_sorted = np.sort(cos, axis=1)
        gamma = np.arccos(cos_sorted[:, -2])
        return np.maximum(gamma, 1e-12)

    def select(self, population, k):
        # No fitness-biased mating selection: each individual already elitist of its subspace.
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        # Random pairing + standard real-coded operators.
        offspring = [deepcopy(ind) for ind in parents]
        random.shuffle(offspring)
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            tools.cxSimulatedBinaryBounded(offspring[i], offspring[i + 1],
                                           eta=self.cx_eta, low=lo, up=hi)  # p_c = 1
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(ind, eta=self.mut_eta, low=lo, up=hi,
                                       indpb=self.mut_prob)
            if ind.fitness.valid:
                del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring          # mu + lambda elitism
        valid = [ind for ind in combined if ind.fitness.valid]
        if len(valid) <= self.pop_size:
            return valid

        F = np.array([ind.fitness.values for ind in valid], dtype=float)
        # Step 1: translate so the ideal is the origin; ||f'|| = distance to ideal.
        z = np.min(F, axis=0)
        self.z_min = z if self.z_min is None else np.minimum(self.z_min, z)
        Fp = F - self.z_min
        dist = np.linalg.norm(Fp, axis=1)
        dist[dist < 1e-12] = 1e-12
        # Step 2: partition by acute angle to the (unit) reference vectors.
        cos = np.clip((Fp / dist[:, None]) @ self.V.T, -1.0, 1.0)
        angle = np.arccos(cos)                     # (n, N)
        assoc = np.argmin(angle, axis=1)
        # Step 3: APD per non-empty subpopulation; keep the min-APD survivor.
        ramp = (self._gen / max(self.max_gen, 1)) ** self.alpha     # (t/t_max)^alpha
        survivors = []
        for j in range(self.pop_size):
            members = np.where(assoc == j)[0]
            if len(members) == 0:
                continue
            theta = angle[members, j]
            penalty = self.n_obj * ramp * (theta / self.gamma[j])
            apd = (1.0 + penalty) * dist[members]
            survivors.append(int(members[int(np.argmin(apd))]))
        # Top up to pop_size from leftovers by smallest distance-to-ideal.
        if len(survivors) < self.pop_size:
            chosen = set(survivors)
            leftover = [i for i in range(len(valid)) if i not in chosen]
            leftover.sort(key=lambda i: dist[i])
            for i in leftover:
                survivors.append(i)
                if len(survivors) >= self.pop_size:
                    break
        return [valid[i] for i in survivors[:self.pop_size]]

    def on_generation(self, gen, population):
        # Remember t for the APD ramp; adapt reference vectors to objective scales.
        self._gen = gen
        period = max(1, int(np.ceil(self.fr * self.max_gen)))
        if gen % period == 0 and population:
            F = np.array([ind.fitness.values for ind in population
                          if ind.fitness.valid], dtype=float)
            if len(F) == 0:
                return
            rng = np.max(F, axis=0) - np.min(F, axis=0)
            rng[rng < 1e-12] = 1e-12
            self.V = self._unit(self.V0 * rng)     # v = v0 ◦ range, renormalized; objectives untouched
            self.gamma = self._gamma(self.V)       # niches moved -> recompute gamma
```
