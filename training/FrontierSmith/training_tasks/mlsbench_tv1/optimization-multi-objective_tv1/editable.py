class CustomMOEA:
    """Failure-regime-first multi-objective strategy scaffold (variant re-aim).

    The score is governed by the worst problem-metric pairing, so this
    scaffold is organised around the two classic failure regimes:
    disconnected true fronts (density machinery bridges the gaps and
    spread collapses) and many-locally-optimal-front landscapes (the
    population commits to the first front it reaches). Both must be
    handled BLIND — the problem identity and its true front are withheld
    — so the only admissible adaptation inputs are online statistics of
    the population's own objective space. Two such signals are computed
    every generation and exposed as continuous quantities:

      * ``self.gap_stat``      — largest normalised nearest-neighbour gap
                                 along the current non-dominated set,
                                 relative to the median gap (large =>
                                 disconnected-looking front);
      * ``self.progress_stat`` — EWMA of the per-generation movement of
                                 the per-objective ideal point (small =>
                                 possible stagnation on a local front).

    An elite archive (``self.archive``) of non-dominated solutions is
    maintained as a hook for stagnation recovery and density-aware
    reinjection. The placeholder machinery is rank+crowding survival that
    uses the statistics only weakly (a graded nudge of mutation
    probability); replace it with machinery that actually solves both
    regimes at once, with one code path for 2 and 3 objectives.

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

        # --- online objective-space statistics (the adaptation channel) ---
        self.gap_stat = 1.0        # ~1-2 on a contiguous, even front
        self.progress_stat = 1.0   # EWMA of ideal-point movement
        self._ideal = None

        # --- elite archive (stagnation-recovery / reinjection hook) ---
        self.archive = []
        self.archive_cap = 2 * pop_size

    # ------------------------------------------------------------------
    # Online self-measurement (problem-blind, continuous)
    # ------------------------------------------------------------------

    def _update_front_stats(self, population: list) -> None:
        """Refresh gap_stat / progress_stat from the current population.

        Uses only the population's own objective vectors: normalised
        nearest-neighbour gaps along the non-dominated set, and the
        movement of the ideal point. No problem identity is consulted.
        """
        front = get_nondominated(population)
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        fmin, fmax = F.min(axis=0), F.max(axis=0)
        span = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
        N = (F - fmin) / span
        if len(N) >= 3:
            d2 = ((N[:, None, :] - N[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(d2.min(axis=1))
            self.gap_stat = float(nn.max() / (np.median(nn) + 1e-12))
        ideal = F.min(axis=0)
        if self._ideal is None:
            self._ideal = ideal
        else:
            new_ideal = np.minimum(self._ideal, ideal)
            delta = float(np.linalg.norm(self._ideal - new_ideal))
            self.progress_stat = 0.8 * self.progress_stat + 0.2 * delta
            self._ideal = new_ideal

    def _update_archive(self, population: list) -> None:
        """Keep a bounded elite archive of non-dominated solutions."""
        merged = self.archive + [deepcopy(ind) for ind in get_nondominated(population)]
        nd = get_nondominated(merged)
        if len(nd) > self.archive_cap:
            compute_crowding_distance(nd)
            nd.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
            nd = nd[: self.archive_cap]
        self.archive = nd

    # ------------------------------------------------------------------
    # Evolutionary operators
    # ------------------------------------------------------------------

    def select(self, population: list, k: int) -> list:
        """Select k parents: binary tournament on rank + crowding.

        Placeholder (rank/crowding); a regime-aware redesign might bias
        mating toward under-covered front segments detected by gap_stat.
        """
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

    def vary(self, parents: list) -> list:
        """SBX + polynomial mutation, with mutation pressure graded by gap_stat.

        The per-variable mutation probability is nudged upward,
        continuously, when the current front looks disconnected — a weak
        placeholder use of the adaptation channel (never an if/else
        between named algorithms).
        """
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        # Continuous nudge: 0 on a contiguous/even front, grows with gaps.
        nudge = max(0.0, min(1.0, (self.gap_stat - 2.0) / 8.0))
        indpb = min(1.0, self.mut_prob * (1.0 + nudge))

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=indpb,
            )
            del ind.fitness.values

        return offspring

    def survive(self, population: list, offspring: list) -> list:
        """Environmental selection from the combined pool.

        Placeholder: non-dominated sorting + crowding-distance truncation.
        This is where both failure regimes are ultimately decided —
        replace with density management that respects gaps and resists
        premature commitment (the archive is available for reinjection).
        """
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)

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
        """Per-generation bookkeeping: refresh statistics and the archive.

        Args:
            gen: current generation number (1-indexed)
            population: population after survival selection
        """
        self._update_front_stats(population)
        self._update_archive(population)
