class CustomMOEA:
    """Evaluation-metered multi-objective strategy scaffold (variant re-aim).

    Under this variant the scarce resource is the number of TRUE objective
    evaluations: the fixed loop only pays for offspring whose fitness the
    ``vary`` step invalidates, so the strategy itself decides, clone by
    clone, which candidates are worth pricing. Each generation's brood is
    split into

      * an ACTIVE subset — actually crossed over / mutated, fitness
        invalidated, hence evaluated and charged to the meter; and
      * FREE RIDERS — untouched clones that keep their parents' valid
        (and still correct) objective vectors and cost nothing.

    ``self.spend_frac`` is the fraction of the brood priced per generation,
    graded continuously by ``on_generation``: while the ideal point is
    still dropping, evaluations are buying convergence and the meter
    opens; when movement fades it closes toward its floor.
    ``self.evals_spent`` is the running ledger of paid evaluations — the
    quantity any efficiency claim must be argued from.

    The machinery around the meter is deliberately ordinary (rank +
    crowding selection, SBX + polynomial variation, duplicate-suppressed
    non-dominated survival). Replace it with machinery that decides WHERE
    to spend, not merely how much: the goal is a final front whose
    quality per paid evaluation embarrasses full-fare baselines, with one
    code path for 2 and 3 objectives.

    Args:
        pop_size: population size
        n_obj: number of objectives
        n_var: number of decision variables
        bounds: (low, high) for all variables
        cx_eta: SBX crossover distribution index (default 20)
        mut_eta: polynomial mutation distribution index (default 20)
        mut_prob: per-variable mutation probability (default 1/n_var)
    """

    def __init__(
        self,
        pop_size: int,
        n_obj: int,
        n_var: int,
        bounds: Tuple[float, float],
        cx_eta: float = 20.0,
        mut_eta: float = 20.0,
        mut_prob: Optional[float] = None,
    ):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

        # --- the meter ---
        self.spend_floor = 0.35    # never price less than this share
        self.spend_ceil = 0.90     # never price more than this share
        self.spend_frac = 0.60     # current share of the brood to price
        self.evals_spent = 0       # ledger: paid objective evaluations

        # --- convergence signal driving the meter ---
        self._ideal_pt = None
        self._gain = 0.05          # EWMA of per-generation ideal-point drop

    def select(self, population: list, k: int) -> list:
        """Rank + crowding binary tournament (placeholder parent choice)."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

    def vary(self, parents: list) -> list:
        """Price only the ACTIVE subset of the brood; the rest ride free.

        Actives are chosen by parent crowding distance (boundary and
        sparse-region parents first, where new information is cheapest).
        Only actives are perturbed and invalidated, so the fixed loop
        evaluates exactly them; untouched clones keep genuinely valid
        inherited fitness and are never re-measured.
        """
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        order = sorted(
            range(len(offspring)),
            key=lambda i: -getattr(offspring[i].fitness, "crowding_dist", 0.0),
        )
        n_active = max(2, int(round(self.spend_frac * len(offspring))))
        n_active = min(len(offspring), n_active + (n_active % 2))
        active = order[:n_active]

        for a, b in zip(active[0::2], active[1::2]):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[a], offspring[b], eta=self.cx_eta, low=lo, up=hi,
                )
        for i in active:
            tools.mutPolynomialBounded(
                offspring[i], eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
            )
            del offspring[i].fitness.values

        self.evals_spent += len(active)
        return offspring

    def survive(self, population: list, offspring: list) -> list:
        """Duplicate-suppressed non-dominated survival.

        Free riding makes exact genotype duplicates common; duplicates
        carry no information and would silently degrade front evenness,
        so the pool is deduplicated before sorting (with a safe fallback
        if too few distinct genotypes remain).
        """
        combined = population + offspring
        seen, pool = set(), []
        for ind in combined:
            key = tuple(ind)
            if key in seen:
                continue
            seen.add(key)
            pool.append(ind)
        if len(pool) < self.pop_size:
            pool = combined

        fronts = tools.sortNondominated(pool, self.pop_size, first_front_only=False)
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break
        return next_gen

    def on_generation(self, gen: int, population: list):
        """Grade the meter from ideal-point movement (open while it pays).

        Args:
            gen: current generation number (1-indexed)
            population: population after survival selection
        """
        front = get_nondominated(population)
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        ideal = F.min(axis=0)
        if self._ideal_pt is None:
            self._ideal_pt = ideal
            return
        new_ideal = np.minimum(self._ideal_pt, ideal)
        drop = float(np.sum(self._ideal_pt - new_ideal))  # L1 improvement >= 0
        self._ideal_pt = new_ideal
        self._gain = 0.7 * self._gain + 0.3 * drop
        span = self.spend_ceil - self.spend_floor
        self.spend_frac = self.spend_floor + span * min(1.0, 12.0 * self._gain)
