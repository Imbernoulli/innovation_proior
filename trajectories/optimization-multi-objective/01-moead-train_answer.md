The baseline I am opening from is the scaffold default, which is already NSGA-II-shaped: dominance-rank tournaments for mating, SBX plus polynomial mutation for variation, and a non-dominated sort with a crowding-distance tie-break for survival. The single thing wrong with it that I want to expose first is that dominance is only a *partial order*. Early in a run that is harmless — most points are clearly bad and get dominated away — but as the population improves, more and more of it becomes mutually non-dominated, and then dominance falls silent: it cannot separate the members of the first front from each other. At that point the only thing deciding survival is the crowding-distance appliance bolted on the side, and the whole thing runs behind an $O(n_{obj}\,N^2)$ sort that, with three objectives where almost every pair is incomparable, carries essentially no information yet still costs a fortune. I want a method whose selection never goes silent and whose diversity falls out for free, and I want to plant it as the floor of the ladder — the cheapest family, conceptually the furthest from the default, with weaknesses that the rest of the ladder is built to repair.

I propose **MOEA/D**: decompose the multi-objective problem into $N$ scalar subproblems, one per evenly spread weight vector, and optimize them cooperatively. The classical fact I lean on is that under mild conditions a Pareto-optimal solution is the optimizer of *some* single scalar aggregation of the objectives — so approximating the front is not one hard vector problem but a *family* of scalar problems, one per weighting, and a scalar problem has a total order. That total order is exactly the selection pressure dominance lacks.

The choice of aggregation is load-bearing. The obvious one is the weighted sum, $\sum_j w_j f_j(x)$, but its level sets are hyperplanes with normal $w$; minimizing slides that hyperplane down until it last touches the attainable set, and on a re-entrant nonconvex stretch the hyperplane skips right over the dent and touches only the two convex shoulders. ZDT3's front here is *disconnected* and the DTLZ fronts are curved, so a weighted sum would systematically miss whole regions — it has the total order but cannot see the whole front. The fix is to change the geometry of the level sets so they can poke into the dents, which is exactly what the **Tchebycheff** aggregation does:

$$g(x \mid w, z) = \max_j\, w_j\,\lvert f_j(x) - z_j\rvert,$$

against the ideal point $z$ with $z_j = \min_x f_j(x)$. Now the level sets are nested axis-aligned right-angle corners (rescaled $L_\infty$ balls) whose vertex marches out along the ray from $z$ in direction $1/w$; because the contour is a corner, not a flat plane, it reaches *into* a nonconvex dent and touches a point no hyperplane could. The optimality argument, not just the picture: take a Pareto-optimal $x^*$ and write the gaps $d_j = f_j(x^*) - z_j$. If every $d_j > 0$, align the corner with $x^*$ by choosing $w_j \propto 1/d_j$, so every product $w_j d_j$ equals the same constant $c$ and $g(x^* \mid w, z) = c$. If some feasible $y$ had $g(y \mid w, z) < c$, then $w_j(f_j(y) - z_j) < c = w_j(f_j(x^*) - z_j)$ for every $j$, hence $f_j(y) < f_j(x^*)$ for all $j$ — $y$ would dominate $x^*$, impossible. So $x^*$ is a minimizer, and *every* front point, convex or not, is reachable. The $\max$ has a kink so $g$ is non-smooth, but this is a derivative-free EA, so I never differentiate it and the kink costs nothing. (The canonical method swaps in boundary-intersection aggregation — PBI, $d_1 + \theta d_2$ — for three or more objectives, where line-projection geometry spreads a 3-objective front more uniformly. This task's edit surface does not expose that and uses plain Tchebycheff throughout; I flag the missing PBI now as a probable weakness on DTLZ2's spherical front.)

Turning the recipe into one population-based run: choose $N$ weight vectors once, spread evenly over the simplex, and let each one *be* a subproblem I optimize and report. For two objectives the even fan is $w = (i/(N-1),\, 1 - i/(N-1))$; for three I need lattice points on the 2-simplex, which the harness supplies through `uniform_reference_points(n_obj, p=12)` — a fixed lattice regardless of the requested `pop_size`, so the effective $N$ is reset to the number of weight vectors the lattice produced. The population *is* the roster of these $N$ current bests, one per subproblem. This is different in kind from the default: there is no global pool to non-dominated-sort, each population slot has an owner (its subproblem) with a total order (its scalar $g$), so selection for that slot never goes silent — and if the weights are evenly spread, the population is organized across the front *by construction*. The diversity the default had to manufacture with crowding distance, decomposition produces as a side effect of the weight distribution.

What makes it work as an EA rather than $N$ isolated hill-climbs is that neighboring subproblems help each other. Because $g$ depends continuously on $w$, two close weight vectors $w^i \approx w^j$ have nearly identical Tchebycheff functions, so the current best of subproblem $j$ is, for free, a *good* candidate for subproblem $i$. So I define for each subproblem $i$ a neighborhood $B(i)$ — the $T$ subproblems whose weights are closest to $w^i$ in Euclidean distance (with $i \in B(i)$) — and the harness fixes $T = 20$. To make progress on subproblem $i$, I draw two parents from $B(i)$, breed one offspring, and try it not just on $i$ but on every neighbor $j \in B(i)$, replacing the current solution of $j$ whenever the offspring scores better on $g(\cdot \mid w^j, z)$. A single good child propagates outward to every nearby subproblem it improves, information flows along the chain of weights, and there is no global sort: each generation is $O(n_{obj}\,N\,T)$ against the default's $O(n_{obj}\,N^2)$, which is why this is the fastest family in the suite.

A few concrete choices. The ideal $z$ cannot be the true $\min_x f_j$ without solving $m$ single-objective problems, so I substitute the running per-objective minimum over every solution evaluated, initialized from the first population and updated whenever a new solution beats $z_j$; Tchebycheff only needs $z$ at or below the front for the corners to open correctly, and the running best converges to that from above. Mating is local with probability $\delta = 0.9$ (parents from $B(i)$), otherwise global (parents from the whole population) — that 10% global jump injects far-away material where local fronts or multimodality make a neighborhood too greedy, exactly DTLZ1's many local fronts. Variation is the shared real-coded pair (SBX $\eta_c = 20$, polynomial mutation $\eta_m = 20$, $p_m = 1/n$), one child per mating, so any difference from the other rungs is the strategy, not the operators. One task-specific detail worth stating: the harness produces offspring positionally and walks $\min(\lvert\text{offspring}\rvert, \lvert\text{weights}\rvert)$ subproblems with an *unbounded* number of neighbor takeovers per child. The canonical method caps how many slots one child may overwrite per generation to protect diversity; this version does not, which is the second place I expect diversity to suffer — on DTLZ2 and the local-front DTLZ1 in particular, where a few good children can clump the population.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 1: MOEA/D
class CustomMOEA:
    """MOEA/D: Multi-Objective Evolutionary Algorithm Based on Decomposition."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.T = 20  # neighborhood size
        self.delta = 0.9  # probability of selecting from neighborhood

        # Generate weight vectors
        self.weights = self._generate_weights(pop_size, n_obj)
        self.pop_size = len(self.weights)  # adjust to actual number of weight vectors

        # Compute neighborhoods
        self.neighbors = self._compute_neighborhoods()

        # Ideal point (updated during search)
        self.z_star = None

    def _generate_weights(self, n, n_obj):
        """Generate uniformly distributed weight vectors."""
        if n_obj == 2:
            weights = []
            for i in range(n):
                w1 = i / max(n - 1, 1)
                weights.append([w1, 1.0 - w1])
            return np.array(weights)
        else:
            # Use DEAP's uniform reference points for 3+ objectives
            ref_points = tools.uniform_reference_points(n_obj, p=12)
            return np.array(ref_points)

    def _compute_neighborhoods(self):
        """Compute T-nearest weight vector neighborhoods."""
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(self.weights, self.weights)
        neighbors = []
        for i in range(len(self.weights)):
            idx = np.argsort(dist_matrix[i])[:self.T]
            neighbors.append(idx.tolist())
        return neighbors

    def _tchebycheff(self, fitness_values, weight, z_star):
        """Tchebycheff scalarization."""
        return max(weight[j] * abs(fitness_values[j] - z_star[j])
                   for j in range(self.n_obj))

    def select(self, population, k):
        """MOEA/D doesn't use standard selection — return population as-is."""
        return [deepcopy(ind) for ind in population]

    def vary(self, parents):
        """Generate one offspring per subproblem using neighborhood mating."""
        offspring = []
        lo, hi = self.bounds

        for i in range(len(parents)):
            # Select mating pool (neighborhood or whole population)
            if random.random() < self.delta:
                pool = [parents[j] for j in self.neighbors[i % len(self.neighbors)]]
            else:
                pool = parents

            # Select two parents from pool
            p1, p2 = random.sample(range(len(pool)), 2)
            child = deepcopy(pool[p1])

            # SBX crossover
            mate = deepcopy(pool[p2])
            if random.random() < 1.0:
                tools.cxSimulatedBinaryBounded(child, mate, eta=self.cx_eta, low=lo, up=hi)

            # Polynomial mutation
            tools.mutPolynomialBounded(child, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del child.fitness.values
            offspring.append(child)

        return offspring

    def survive(self, population, offspring):
        """MOEA/D survival: update subproblems using Tchebycheff decomposition."""
        # Update ideal point
        all_inds = [ind for ind in population + offspring if ind.fitness.valid]
        if not all_inds:
            return population

        if self.z_star is None:
            self.z_star = [float('inf')] * self.n_obj
        for ind in all_inds:
            for j in range(self.n_obj):
                if ind.fitness.values[j] < self.z_star[j]:
                    self.z_star[j] = ind.fitness.values[j]

        # Update each subproblem
        next_gen = list(population)
        for i in range(min(len(offspring), len(self.weights))):
            child = offspring[i]
            if not child.fitness.valid:
                continue

            # Update neighbors
            neighbors_idx = self.neighbors[i % len(self.neighbors)]
            for j_idx in neighbors_idx:
                if j_idx >= len(next_gen):
                    continue
                g_child = self._tchebycheff(child.fitness.values, self.weights[j_idx], self.z_star)
                g_current = self._tchebycheff(next_gen[j_idx].fitness.values, self.weights[j_idx], self.z_star)
                if g_child < g_current:
                    next_gen[j_idx] = deepcopy(child)

        return next_gen[:self.pop_size]

    def on_generation(self, gen, population):
        pass
```
