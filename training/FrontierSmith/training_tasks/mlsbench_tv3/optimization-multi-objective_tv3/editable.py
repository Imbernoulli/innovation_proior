class CustomMOEA:
    """Evenness-first multi-objective strategy scaffold (variant re-aim).

    The evenness number is computed differently per dimensionality — for
    two objectives it is the mean absolute deviation of consecutive gaps
    along the f1-sorted front; for three, the dispersion of nearest-
    neighbour distances — and this scaffold makes that number the
    organising concern while refusing to give back convergence. Two
    mechanisms carry the re-aim:

      * GAP-SEEKING MATING: each generation the widest hole in the
        current non-dominated set is located (consecutive-gap scan in
        2-D, most-isolated-member scan in 3-D) and the two solutions
        flanking it are planted at the head of the parent list, so
        pairwise SBX recombines exactly across the hole.
      * UNIFORMITY TRUNCATION: when the last front must be cut, members
        are not ranked by crowding; instead the individual with the
        smallest nearest-neighbour distance in normalised objective
        space is deleted repeatedly (per-objective extremes protected),
        a direct descent on the dispersion the scorer reads.

    ``self.evenness_log`` records the internal evenness statistic per
    generation so the discipline can be argued from a trace. Variation is
    stock SBX + polynomial mutation and convergence pressure is stock
    non-dominated sorting; replace anything, but every replacement must
    answer to both books — the hole must close without the dominated
    volume shrinking.

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

        # Per-generation evenness trace: list of (gen, statistic).
        self.evenness_log = []

    # ------------------------------------------------------------------
    # Geometry helpers (objective space, normalised per call)
    # ------------------------------------------------------------------

    def _normed(self, front: list) -> np.ndarray:
        """Front objective matrix scaled to [0, 1] per objective."""
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        fmin, fmax = F.min(axis=0), F.max(axis=0)
        span = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
        return (F - fmin) / span

    def _gap_flankers(self, front: list) -> list:
        """Indices of the two members flanking the widest hole."""
        if len(front) < 3:
            return list(range(len(front)))
        N = self._normed(front)
        if self.n_obj == 2:
            order = np.argsort(N[:, 0])
            S = N[order]
            gaps = np.sqrt(((S[1:] - S[:-1]) ** 2).sum(axis=1))
            g = int(np.argmax(gaps))
            return [int(order[g]), int(order[g + 1])]
        d2 = ((N[:, None, :] - N[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        iso = int(np.argmax(d2.min(axis=1)))
        mate = int(np.argmin(d2[iso]))
        return [iso, mate]

    def _truncate_uniform(self, front: list, k: int) -> list:
        """Delete min-nearest-neighbour members until k remain."""
        N = self._normed(front)
        alive = list(range(len(front)))
        protected = {int(np.argmin(N[:, m])) for m in range(N.shape[1])}
        while len(alive) > k:
            sub = N[alive]
            d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            order = np.argsort(d2.min(axis=1))
            victim = None
            for idx in order:
                if alive[int(idx)] not in protected:
                    victim = int(idx)
                    break
            if victim is None:
                victim = int(order[0])
            alive.pop(victim)
        return [front[i] for i in alive]

    # ------------------------------------------------------------------
    # Evolutionary operators
    # ------------------------------------------------------------------

    def select(self, population: list, k: int) -> list:
        """DCD tournament, with the widest-hole flankers planted first.

        Positions 0 and 1 of the returned parents flank the largest gap
        of the first front, so the first SBX pair recombines across it.
        """
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        chosen = tools.selTournamentDCD(population, k)
        flank = [fronts[0][i] for i in self._gap_flankers(fronts[0])]
        for j, ind in enumerate(flank[: len(chosen)]):
            chosen[j] = ind
        return chosen

    def vary(self, parents: list) -> list:
        """Stock SBX + polynomial mutation (all offspring re-evaluated)."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
            )
            del ind.fitness.values
        return offspring

    def survive(self, population: list, offspring: list) -> list:
        """Rank-first survival with uniformity truncation of the cut front."""
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                next_gen.extend(self._truncate_uniform(front, remaining))
                break
        return next_gen

    def on_generation(self, gen: int, population: list):
        """Record the internal evenness statistic for the discipline trace.

        Args:
            gen: current generation number (1-indexed)
            population: population after survival selection
        """
        front = get_nondominated(population)
        if len(front) < 3:
            return
        N = self._normed(front)
        if self.n_obj == 2:
            S = N[np.argsort(N[:, 0])]
            gaps = np.sqrt(((S[1:] - S[:-1]) ** 2).sum(axis=1))
            mean_gap = float(gaps.mean()) + 1e-12
            stat = float(np.abs(gaps - gaps.mean()).sum() / (len(gaps) * mean_gap))
        else:
            d2 = ((N[:, None, :] - N[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d2, np.inf)
            nn = np.sqrt(d2.min(axis=1))
            stat = float(nn.std() / (nn.mean() + 1e-12))
        self.evenness_log.append((gen, stat))
