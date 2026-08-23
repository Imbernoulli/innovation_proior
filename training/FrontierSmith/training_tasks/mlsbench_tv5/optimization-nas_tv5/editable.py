class NASOptimizer:
    """Single-trajectory local-search NAS scaffold (variant re-aim).

    Policy restriction, enforced by construction: ONE random draw seeds
    the walk, and every later oracle call evaluates a single-edit
    neighbour of the current position. No restarts, no global sampling,
    no portfolios — if the walk stalls, the remedy must be smarter move
    selection, not teleportation. State:

      * ``self.position`` / ``self.position_acc`` — where the walk
        stands and what that cell scored;
      * ``self.visited`` — cells already evaluated: a neighbour is never
        paid for twice, and when a 24-cell neighbourhood is exhausted
        the fallback re-centres on the best evaluated cell along the
        trail (still a trajectory move, not a restart);
      * ``self.trail`` — the ordered move log, the exhibit proving every
        step obeyed the single-trajectory contract;
      * acceptance — move whenever the neighbour is at least as good up
        to ``self.drift_tol`` (sideways drift across plateaus).

    The placeholder picks neighbours uniformly; the invited redesign is
    move ORDER and acceptance — edge-priority schedules informed by
    edits that paid off, first-improve versus best-improve sweeps,
    plateau escape rules — all inside the contract. The returned cell is
    the best ever probed, kept as a free by-product of the walk.
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

        self.position = None       # current cell of the walk
        self.position_acc = -1.0
        self.visited = set()       # cells already paid for
        self.trail = []            # ordered (arch, val_acc) move log
        self.drift_tol = 0.0       # sideways-drift tolerance, in points

        self.best_arch = None
        self.best_val_acc = -1.0

    def _unvisited_neighbor(self):
        """A not-yet-paid single-edit move from the current position."""
        nbrs = [n for n in get_neighbors(self.position)
                if is_valid_arch(n) and tuple(n) not in self.visited]
        if nbrs:
            return random.choice(nbrs)
        # Neighbourhood exhausted: re-centre on the best evaluated cell
        # along the trail and draw again — a move, not a restart.
        for a, _acc in sorted(self.trail, key=lambda t: -t[1]):
            if a != self.position:
                self.position = list(a)
                break
        nbrs = [n for n in get_neighbors(self.position)
                if is_valid_arch(n) and tuple(n) not in self.visited]
        if nbrs:
            return random.choice(nbrs)
        return mutate_architecture(self.position)

    def search_step(self, epoch):
        """Advance the walk by one paid probe.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        if self.position is None:
            arch = random_architecture()   # the single global draw
        else:
            arch = self._unvisited_neighbor()

        val_acc = self.api.query_val_accuracy(arch)
        self.visited.add(tuple(arch))
        self.trail.append((arch, val_acc))

        # Acceptance: step onto the probed cell if it is not worse than
        # the position by more than the drift tolerance.
        if self.position is None or val_acc >= self.position_acc - self.drift_tol:
            self.position = arch
            self.position_acc = val_acc

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Best cell ever probed — remembered for free along the trail."""
        return self.best_arch
