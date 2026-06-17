**Problem.** Both prior rungs do convergence through dominance (default, AGE-MOEA) or a flat-calibrated
decomposition (MOEA/D), then bolt diversity on top. Dominance is a partial order that goes blind as objectives
grow — the first front swallows the population and a density heuristic is left doing the convergence job. That
is why AGE-MOEA converged *short* on DTLZ2 (HV 2.572, below decomposition's 2.737): its diversity-only trim
spread points across a sphere that hadn't fully converged.

**Key idea.** Drop dominance from selection entirely; use a total order per reference direction. Translate
objectives so the running ideal is the origin, so `||f'||` is the convergence reading and the *angle* of `f'`
to a unit reference vector is a bounded, scale-invariant diversity reading. Associate each individual with its
nearest reference vector and keep, per niche, the individual of minimum angle-penalized distance
`APD = ||f'|| · (1 + alpha·(t/t_max)^2 · angle)`. The time-ramp gives near-pure convergence early and spread
late — restoring the convergence-in-front pressure AGE-MOEA lacked.

**Why (and what this harness simplifies).** The APD here is a *simplified* form: it keeps the `(t/t_max)^2`
ramp (`alpha = 2`) but drops the canonical per-niche angle normalization (`θ/γ`) and the dimensional `M`
factor, using the raw associated angle. Reference vectors are adapted to the objective ranges (not the
objectives normalized, which would corrupt the angle/length geometry) every `fr·max_gen` generations. So
DTLZ2 convergence should recover, while the un-normalized angle penalty leaves spread slightly under-tuned —
especially on the local-front DTLZ1 and under the hard-wired `max_gen = 400` horizon.

**Hyperparameters.** APD penalty `alpha = 2`; adaptation frequency `fr = 0.1`; `max_gen = 400` (ramp
denominator + adaptation period); reference points `p = pop_size-1` (2-obj) / `p = 12` (3-obj); SBX
`eta_c = 20` (always applied), polynomial mutation `eta_m = 20`, `p_m = 1/n_var`.

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
