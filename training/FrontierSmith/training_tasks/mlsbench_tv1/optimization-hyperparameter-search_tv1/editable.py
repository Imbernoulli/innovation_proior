class CustomHPOStrategy:
    """Anytime-first HPO strategy scaffold (variant re-aim).

    The run is judged jointly, on every benchmark, by the final incumbent
    AND by the area under the incumbent-versus-cost curve: the strategy
    must reach near-final quality with a small fraction of the spent
    budget without giving the endpoint back. Cost is charged as the sum
    of fidelities, so fidelity is the main lever — cheap low-fidelity
    screens buy early curve area, but their scores are biased and noisy
    and must be confirmed at full fidelity before they can be trusted.

    Structural hooks provided by this scaffold (redesign freely):
      * a declared budget split (``SCREEN_COST_FRAC``) between a
        low-fidelity screening phase and a full-fidelity phase;
      * ``_fidelity_schedule(spent_frac)`` mapping budget progress to the
        fidelity of the next evaluation;
      * an anytime trace (``self.trace``) recording, at every suggestion,
        (cost spent, best full-fidelity score so far) — the run's own
        record of the anytime-vs-final trade-off, usable for adaptation;
      * promotion of the best screened configuration to a full-fidelity
        confirmation at the phase boundary (cross-fidelity comparisons
        are treated as hypotheses to confirm, never as final answers).

    One configuration serves all three benchmarks (mixed float/int/
    categorical parameters, log scales, 3-6 dims): no benchmark-specific
    branching. The placeholder policy is deliberately weak — random
    screening at low fidelity, a single promotion, then random search at
    full fidelity.

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: float in (0, 1] — fraction of a full evaluation;
                  the loop clips it to [0.1, 1.0] and charges it.
    """

    SCREEN_COST_FRAC = 0.10   # fraction of total cost spent on screening
    SCREEN_FIDELITY = 0.50    # fidelity used during the screening phase

    def __init__(self, seed: int = 42):
        """Initialize the strategy.

        Stores the seed, creates the RNG, and initialises the anytime
        bookkeeping. The total budget is learned on the first call to
        suggest() from history + budget_left.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.total_budget = None     # learned on first suggest()
        self.trace = []              # (spent_cost, best_full_fidelity_score)
        self._promoted = False       # phase-boundary confirmation done?

    def _spent(self, history: List[Trial]) -> float:
        """Cost charged so far (sum of fidelities over past trials)."""
        return float(sum(t.budget for t in history))

    def _fidelity_schedule(self, spent_frac: float) -> float:
        """Budget-split fidelity policy: screen cheap, then confirm at 1.0."""
        if spent_frac < self.SCREEN_COST_FRAC:
            return self.SCREEN_FIDELITY
        return 1.0

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Propose the next configuration and its evaluation fidelity.

        Placeholder policy (weak on purpose): uniform random proposals,
        with fidelity set by the declared budget split; at the boundary
        between phases the best screened configuration is re-evaluated
        once at full fidelity so a cheap score can become a trusted
        incumbent.
        """
        if self.total_budget is None:
            self.total_budget = self._spent(history) + float(budget_left)
        spent = self._spent(history)
        spent_frac = spent / max(self.total_budget, 1e-9)

        # Anytime trace: best score observed at FULL fidelity so far.
        full = [t.score for t in history if t.budget >= 0.999]
        best_full = max(full) if full else float("-inf")
        self.trace.append((spent, best_full))

        fidelity = self._fidelity_schedule(spent_frac)

        # Phase boundary: confirm the best screened config at full fidelity.
        if fidelity >= 1.0 and not self._promoted:
            self._promoted = True
            screened = [t for t in history if t.budget < 0.999]
            if screened:
                best_screen = max(screened, key=lambda t: t.score)
                return dict(best_screen.config), 1.0

        config = space.sample_uniform(self.rng)
        return config, fidelity
