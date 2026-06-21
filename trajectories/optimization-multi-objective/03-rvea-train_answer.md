AGE-MOEA landed exactly on the prediction. On spread it did its job — DTLZ2 dropped to 0.519 (from decomposition's 0.846) and DTLZ1 to mean 2.16 (from 4.41) — but on DTLZ2 *convergence regressed*: HV mean 2.572 against decomposition's 2.737, IGD 0.0905 against 0.0761, both worse. That is the diagnosis I wrote in advance: keeping only the diversity half of the survival score, with no proximity-to-ideal term, spreads well-separated points across a sphere that has not fully converged. The deeper fact behind it is that AGE-MOEA — like the default NSGA-II — still does its primary convergence work through Pareto dominance, then bolts diversity on top. Dominance is a partial order, and a partial order goes blind exactly when everything is incomparable: as objective count rises, the chance one random vector beats another on *every* coordinate falls, so almost every pair becomes mutually non-dominated, the first front swallows the population early, and a density heuristic is left doing the convergence job it was never meant for. So the rung I want now abandons dominance for selection entirely and replaces it with a *total order per direction* that fuses convergence and diversity into a single scalar, so selection never goes blind and convergence is never delegated to a density tie-break.

I propose **RVEA**, the reference-vector guided approach with an angle-penalized distance. The structural improvement over the decomposition rung's single Tchebycheff corner-distance is to split the scalar into two genuinely orthogonal readings — *length* for convergence and *angle* for diversity. First translate the objectives so the running ideal point (per-pool per-objective minimum $z_{\min}$) sits at the origin: $f' = f - z_{\min}$. This pins the ideal to the origin, puts every $f'$ in the first quadrant, and makes $\lVert f'\rVert$ the distance to the ideal — a clean convergence reading, for free. For diversity I want the *direction mismatch* between a solution and a prescribed search direction, independent of how far out the solution sits. That is an angle, and an angle is exactly what a Euclidean off-line distance (the PBI-style penalty) is not: it is bounded in $[0, \pi/2]$ for first-quadrant vectors and *scale-invariant*, because $\cos\theta = (a\cdot b)/(\lVert a\rVert\,\lVert b\rVert)$ is homogeneous of degree zero — scaling a solution by any positive constant leaves its angle to a reference direction unchanged. Length and angle are decoupled: convergence and diversity, read separately.

The directions are a fan of reference vectors built from `uniform_reference_points` — $p = \text{pop\_size}-1$ for two objectives, $p = 12$ for three — each normalized to unit length so cosines need no renormalization on the vector side, with a pristine copy stashed for later adaptation. Each generation, after combining parents and offspring (the $\mu+\lambda$ shell), translate by $z_{\min}$, normalize each $f'$ to unit length, take the cosine matrix $F_{\text{normalized}} @ \text{ref\_vectors}^\top$, convert to angles, and associate each individual with the reference vector of minimum angle. That carves the pool into subpopulations, one per vector; from each non-empty subpopulation keep the single individual of minimum angle-penalized distance. No dominance comparison appears anywhere — selection is a total order within each subspace, so it cannot go blind.

The scalar inside a subpopulation is the **angle-penalized distance (APD)**, and I must be exact about what this harness implements versus the canonical form, because the difference is load-bearing. The full APD measures the angle *relative to the niche's own width* ($\theta/\gamma_{v_j}$, where $\gamma$ is the nearest-neighbor angle of reference vector $v_j$) and multiplies by the objective count $M$ to keep the penalty meaningful as objective space gets sparser. This edit surface does *neither*. Its penalty is simply

$$\gamma_t = \alpha\,(t/t_{\max})^2, \qquad \text{APD} = \lVert f'\rVert \cdot \big(1 + \gamma_t\cdot\theta_{\min}\big),$$

with the *raw* associated angle $\theta_{\min}$, $\alpha = 2$, and no per-niche $\gamma$ normalization, no $M$ factor. The time-ramp is the part that matters most, and it is kept: early in the run $(t/t_{\max})^2 \approx 0$ so APD $\approx \lVert f'\rVert$ — pure convergence pressure — and late in the run the angle penalty bites, so the method shifts from converge-first to spread-later on its own. That is precisely the missing convergence-in-front pressure. AGE-MOEA's survival had *no* convergence pressure inside a front; the APD, by contrast, is fundamentally a *scaled convergence distance* — even at full penalty it is $(1 + \text{penalty})\cdot\lVert f'\rVert$, so the closer-to-front solution always has the edge unless its angle is badly off, and early in the run the penalty is near zero so the algorithm races each niche straight onto the front before it ever worries about spread. So on DTLZ2 I expect HV to recover past AGE-MOEA's 2.572, plausibly back above decomposition's 2.737. The cost of dropping niche-width normalization: niches of different angular spacing are penalized inconsistently, so on a problem with an irregular reference fan the spread should come out a little less even than a fully-normalized APD would manage — and on the local-front DTLZ1 the un-normalized angle penalty is a recipe for an occasional blow-up seed.

Two more pieces. Variation is deliberately plain — each individual is already the elitist of its own subspace, so there is no reason to bias mating; `select` just shuffles the population and returns a slice, and `vary` does random-pair SBX ($\eta_c = 20$, always applied) plus polynomial mutation ($\eta_m = 20$, $p_m = 1/n$). All the convergence/diversity balancing lives in `survive`. Reference-vector adaptation handles badly-scaled fronts *without normalizing the objectives* — which would be poison, since the APD reads the actual objective values and normalizing would mangle the angle-and-length geometry every generation. Instead `on_generation` records the current generation for the ramp and, every $\lfloor f_r\cdot t_{\max}\rfloor$ generations with $f_r = 0.1$, stretches the pristine reference vectors by the population's per-objective range $z_{\max} - z_{\min}$ and renormalizes, tilting the fan toward the front's long axes while leaving the objectives untouched. The harness hard-wires $t_{\max} = 400$ for both the ramp denominator and the adaptation period and defaults the ramp's generation estimate to $t_{\max}/2$ before the first `on_generation` fires; that constant is task-specific and means on the shorter-horizon ZDT runs (200 generations) the ramp only reaches $(200/400)^2 = 0.25$ of full penalty, so the spread pressure on the ZDTs is gentler than on the DTLZ runs — a quirk that should slightly favor convergence over spread on ZDT1/ZDT3. My overall expectation: RVEA clears the two prior rungs on aggregate by fixing the convergence wound, but its un-normalized APD and hard-wired horizon leave enough diversity slack that the cleaner-spread methods above it can still beat it.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 3: RVEA
class CustomMOEA:
    """RVEA: Reference Vector Guided Evolutionary Algorithm."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.alpha = 2.0  # penalty parameter for APD
        self.fr = 0.1  # frequency of reference vector adaptation

        # Generate initial reference vectors
        if n_obj == 2:
            p = pop_size - 1
            self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=p))
        else:
            self.ref_vectors = np.array(tools.uniform_reference_points(n_obj, p=12))
        self.ref_vectors_initial = self.ref_vectors.copy()

        # Normalize reference vectors to unit length
        norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        self.ref_vectors = self.ref_vectors / norms

    def _angle_penalized_distance(self, fitness_values, gen, max_gen):
        """Compute angle-penalized distance for each individual to its closest reference vector."""
        F = np.array(fitness_values)
        n = len(F)
        n_ref = len(self.ref_vectors)

        if n == 0:
            return np.array([]), np.array([])

        # Translate objectives (subtract ideal point)
        z_min = np.min(F, axis=0)
        F_translated = F - z_min + 1e-12

        # Compute angles between each individual and each reference vector
        F_norms = np.linalg.norm(F_translated, axis=1, keepdims=True)
        F_norms[F_norms < 1e-12] = 1e-12
        F_normalized = F_translated / F_norms

        # Cosine similarity
        cos_angles = F_normalized @ self.ref_vectors.T  # (n, n_ref)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)  # (n, n_ref)

        # Associate each individual with closest reference vector
        associations = np.argmin(angles, axis=1)  # (n,)
        min_angles = angles[np.arange(n), associations]  # (n,)

        # Compute convergence (distance along reference vector)
        convergence = F_norms.flatten()

        # Angle penalty that increases over generations
        gamma = self.alpha * (gen / max(max_gen, 1)) ** 2

        # APD = convergence * (1 + gamma * angle)
        apd = convergence * (1.0 + gamma * min_angles)

        return apd, associations

    def select(self, population, k):
        """Random mating selection."""
        selected = [deepcopy(ind) for ind in population]
        random.shuffle(selected)
        return selected[:k]

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 1.0:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population, offspring):
        """RVEA survival: angle-penalized distance based selection."""
        combined = population + offspring
        valid = [ind for ind in combined if ind.fitness.valid]

        if len(valid) <= self.pop_size:
            return valid

        fitness_values = [ind.fitness.values for ind in valid]
        # Use a large gen estimate based on problem config
        max_gen = 400
        gen_estimate = getattr(self, '_current_gen', max_gen // 2)
        apd, associations = self._angle_penalized_distance(fitness_values, gen_estimate, max_gen)

        # Select the best individual per reference vector (lowest APD)
        selected_indices = set()
        n_ref = len(self.ref_vectors)
        for v in range(n_ref):
            mask = np.where(associations == v)[0]
            if len(mask) > 0:
                best_idx = mask[np.argmin(apd[mask])]
                selected_indices.add(best_idx)

        # If not enough, fill with best remaining by APD
        if len(selected_indices) < self.pop_size:
            remaining = [i for i in range(len(valid)) if i not in selected_indices]
            remaining.sort(key=lambda i: apd[i])
            for i in remaining:
                selected_indices.add(i)
                if len(selected_indices) >= self.pop_size:
                    break

        # If too many (more ref vectors than pop_size), truncate by APD
        selected_list = sorted(selected_indices, key=lambda i: apd[i])[:self.pop_size]
        return [valid[i] for i in selected_list]

    def on_generation(self, gen, population):
        """Adapt reference vectors periodically."""
        self._current_gen = gen

        # Reference vector adaptation
        max_gen = 400
        if gen % max(1, int(self.fr * max_gen)) == 0 and len(population) > 0:
            fitness_values = np.array([ind.fitness.values for ind in population if ind.fitness.valid])
            if len(fitness_values) > 0:
                z_max = np.max(fitness_values, axis=0)
                z_min = np.min(fitness_values, axis=0)
                scale = z_max - z_min
                scale[scale < 1e-12] = 1.0

                # Scale reference vectors
                self.ref_vectors = self.ref_vectors_initial * scale
                norms = np.linalg.norm(self.ref_vectors, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1e-12
                self.ref_vectors = self.ref_vectors / norms
```
