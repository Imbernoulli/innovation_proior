class CustomMOEA:
    """Frozen-configuration multi-objective strategy scaffold (variant re-aim).

    Nothing in this class adapts. Every control quantity is decided in
    the FROZEN block below, before the first individual is created, and
    is shared by every problem instance in the suite at both objective
    counts. The experiment is whether a single, structurally sound,
    static recipe can carry the whole benchmark — so any hidden feedback
    loop (a probability that drifts, a schedule indexed by generation, a
    stagnation detector) would spoil the evidence and is out of bounds.

    The one static design liberty the placeholder takes is a fixed
    VARIATION PORTFOLIO: most offspring receive fine-grained polynomial
    mutation (high distribution index), a constant minority receive a
    coarse, long-range one (low index). The shares never change.
    Redesign the portfolio, the operators, or the survival rule freely —
    but keep the contract: constants only, chosen once, identical
    everywhere; ``on_generation`` stays inert and no attribute is
    written after ``__init__``.

    FROZEN constants (class attributes, set once):
        CX_PROB       — SBX probability per parent pair.
        FINE_ETA      — distribution index of the fine mutation.
        COARSE_ETA    — distribution index of the coarse mutation.
        COARSE_SHARE  — fixed fraction of offspring mutated coarsely.
        MUT_SCALE     — multiplier on the 1/n_var per-gene rate.

    Args:
        pop_size: population size
        n_obj: number of objectives
        n_var: number of decision variables
        bounds: (low, high) for all variables
        cx_eta: SBX crossover distribution index (default 20)
        mut_eta: polynomial mutation distribution index (default 20)
        mut_prob: per-variable mutation probability (default 1/n_var)
    """

    CX_PROB = 0.9
    FINE_ETA = 20.0
    COARSE_ETA = 4.0
    COARSE_SHARE = 0.15
    MUT_SCALE = 1.0

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
        # No adaptive state. Nothing below ever writes an attribute.

    def select(self, population: list, k: int) -> list:
        """Binary tournament on rank + crowding, with fixed pressure.

        Args:
            population: current population (list of Individuals)
            k: number of parents to select
        Returns:
            list of k selected individuals
        """
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

    def vary(self, parents: list) -> list:
        """Fixed-portfolio variation: SBX, then fine-or-coarse mutation.

        Each offspring draws its mutation intensity from the constant
        portfolio (COARSE_SHARE coarse, the rest fine). The draw is
        random but the distribution over intensities never moves — that
        is the difference between a portfolio and an adaptation.
        """
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.CX_PROB:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
        indpb = min(1.0, self.MUT_SCALE * self.mut_prob)
        for ind in offspring:
            eta = self.COARSE_ETA if random.random() < self.COARSE_SHARE else self.FINE_ETA
            tools.mutPolynomialBounded(ind, eta=eta, low=lo, up=hi, indpb=indpb)
            del ind.fitness.values
        return offspring

    def survive(self, population: list, offspring: list) -> list:
        """Static environmental selection: rank, then crowding truncation.

        Args:
            population: current population
            offspring: newly generated offspring
        Returns:
            list of pop_size individuals for the next generation
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
        """Deliberately inert — the no-adaptation contract of this variant.

        Args:
            gen: current generation number (1-indexed)
            population: current population after survival selection
        """
        pass
