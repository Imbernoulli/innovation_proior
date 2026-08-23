class CustomMOEA:
    """Boundary-anchored, volume-weighted strategy scaffold (variant re-aim).

    Dominated volume is not spread evenly along a front: with the
    scorer's reference point sitting just outside the objective ranges,
    the objective-wise extreme members pin slabs of volume no interior
    point can recover, and each member's marginal worth is its EXCLUSIVE
    contribution — the region only it dominates. This scaffold books
    survival credit in those terms:

      * ANCHORS — the per-objective minimisers of the front being cut
        are seated first, unconditionally, every generation;
      * CONTRIBUTION CREDIT (2-D) — remaining members of the cut front
        are ranked by their exact exclusive-rectangle area (from sorted
        neighbours) and kept greedily;
      * in 3-D the placeholder falls back to crowding distance as the
        contribution proxy — an honest gap left open for redesign.

    Mating follows the same book: parents come from a stock dominance
    tournament, but the current anchors are re-seated at the tail of the
    parent list so the boundary keeps being explored outward (the final
    SBX pair recombines two anchors). The degeneracy check is explicit:
    evenness is still scored, so ``self.extent_log`` tracks per-objective
    front spans per generation — extent gains must be distinguishable
    from interior collapse when the claim is argued.

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

        # Per-generation record of the front's per-objective spans.
        self.extent_log = []

    # ------------------------------------------------------------------
    # Volume accounting
    # ------------------------------------------------------------------

    def _anchor_indices(self, F: np.ndarray) -> list:
        """Indices of the per-objective minimisers (deduplicated)."""
        anchors = []
        for m in range(F.shape[1]):
            a = int(np.argmin(F[:, m]))
            if a not in anchors:
                anchors.append(a)
        return anchors

    def _credit_cut(self, front: list, k: int) -> list:
        """Anchors first, then the largest exclusive contributions."""
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        anchors = self._anchor_indices(F)[:k]

        if self.n_obj == 2 and len(front) >= 3:
            order = np.argsort(F[:, 0])
            S = F[order]
            credit = np.full(len(front), np.inf)
            for j in range(1, len(order) - 1):
                area = (S[j + 1, 0] - S[j, 0]) * (S[j - 1, 1] - S[j, 1])
                credit[int(order[j])] = area
        else:
            compute_crowding_distance(front)
            credit = np.array(
                [getattr(ind.fitness, "crowding_dist", 0.0) for ind in front]
            )

        chosen = list(anchors)
        for idx in np.argsort(-credit):
            if len(chosen) >= k:
                break
            if int(idx) not in chosen:
                chosen.append(int(idx))
        return [front[i] for i in chosen]

    # ------------------------------------------------------------------
    # Evolutionary operators
    # ------------------------------------------------------------------

    def select(self, population: list, k: int) -> list:
        """Dominance tournament with anchors re-seated at the tail."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        chosen = tools.selTournamentDCD(population, k)
        F = np.array([ind.fitness.values for ind in fronts[0]], dtype=float)
        seats = [fronts[0][i] for i in self._anchor_indices(F)]
        for j, ind in enumerate(seats[: len(chosen)]):
            chosen[-(j + 1)] = ind
        return chosen

    def vary(self, parents: list) -> list:
        """SBX + polynomial mutation over the anchor-seeded parent list."""
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
        """Rank-first survival; the cut front is settled by volume credit."""
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)
        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                next_gen.extend(self._credit_cut(front, remaining))
                break
        return next_gen

    def on_generation(self, gen: int, population: list):
        """Track front extent so boundary gains can be audited.

        Args:
            gen: current generation number (1-indexed)
            population: population after survival selection
        """
        front = get_nondominated(population)
        F = np.array([ind.fitness.values for ind in front], dtype=float)
        self.extent_log.append((gen, (F.max(axis=0) - F.min(axis=0)).tolist()))
