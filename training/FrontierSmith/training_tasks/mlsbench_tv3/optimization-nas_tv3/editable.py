class NASOptimizer:
    """Anytime-incumbent NAS scaffold (variant re-aim).

    Discipline: after EVERY paid query the optimizer must be ready to be
    stopped — the loop can end at any epoch (budget exhaustion aborts a
    step in flight) and whatever ``get_best_architecture`` returns at
    that instant is what gets tested. Consequences baked in:

      * the incumbent is refreshed inside ``search_step``, immediately
        after the query returns — never in a deferred consolidation;
      * ``get_best_architecture`` is a plain read of the incumbent; it
        does no work, because at stop time there may be no budget left
        to do any;
      * improvement opportunities are INTERLEAVED, not phased — fresh
        coverage cells and single-edit refinements of the incumbent
        alternate epoch by epoch, so no stretch of the run is a pure
        setup phase whose payoff arrives only later;
      * ``self.incumbent_curve`` records (epoch, best_val_acc) after
        every step: the anytime profile from which any claim about
        truncation-robustness has to be argued.

    The placeholder alternation is rigid; the invited redesign is a
    schedule that still improves the incumbent early and often (banded
    exploration, laddered refinement, adaptive interleaving) without
    ever hollowing out the guarantee that stopping now is safe.
    """

    def __init__(self, api, num_epochs, seed):
        """Initialize the optimizer.

        Args:
            api: BenchmarkAPI (with budget = num_epochs validation queries).
            num_epochs: Total number of allowed validation queries (budget).
            seed: Random seed for reproducibility.
        """
        self.api = api
        self.num_epochs = num_epochs
        self.seed = seed

        self._tried = set()          # avoid paying twice for one cell
        self.incumbent_curve = []    # (epoch, best_val_acc) after each step

        self.best_arch = None
        self.best_val_acc = -1.0

    def _next_candidate(self, epoch):
        """Interleave coverage (even epochs) with refinement (odd epochs)."""
        for _ in range(15):
            if self.best_arch is None or epoch % 2 == 0:
                cand = random_architecture()
            else:
                cand = mutate_architecture(self.best_arch)
            if is_valid_arch(cand) and tuple(cand) not in self._tried:
                return cand
        return random_architecture()

    def search_step(self, epoch):
        """Query once, refresh the incumbent at once, log the curve point.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        arch = self._next_candidate(epoch)
        val_acc = self.api.query_val_accuracy(arch)
        self._tried.add(tuple(arch))

        # Incumbent refresh happens HERE, before anything else can fail:
        # from this line on, stopping the run is safe.
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch
        self.incumbent_curve.append((epoch, self.best_val_acc))

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Zero-work read: the incumbent is always commit-ready."""
        return self.best_arch
