class NASOptimizer:
    """Risk-averse, seed-stable NAS search scaffold (variant re-aim).

    The scoreboard is the TEST accuracy of the single returned cell,
    averaged over seeds and combined across datasets by a geometric mean:
    the target is the floor of the outcome distribution, not its mean. A
    strategy is only as good as its worst seed on its worst dataset, and
    the final commitment is judged on a split it never queried. This
    scaffold therefore separates three concerns:

      * a declared exploration/refinement budget split
        (``self.explore_frac`` of the queries go to coverage of the
        space, the rest near incumbents) so no single early misstep can
        strand a run;
      * a full query ledger (``self.history``) of (arch, val_acc) pairs —
        the raw material for any evidence-based final choice;
      * ``_evidence_score(arch, val_acc)`` — the commitment criterion:
        instead of trusting the raw validation argmax (the candidate most
        exposed to a lucky number), it corroborates a candidate with the
        queried accuracies of its 1-edit neighbours;
        ``get_best_architecture`` returns the ledger entry with the best
        evidence score.

    The placeholder policy is deliberately weak: random coverage, then
    1-edge mutations of the incumbent, with a mild evidence blend.

    Budget discipline: every call to ``api.query_val_accuracy`` counts,
    repeats included; the harness aborts the run on overrun, so the
    ledger is also used to avoid paying twice for the same architecture.
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

        # Budget split: fraction of queries spent on space coverage
        # before refinement around incumbents begins.
        self.explore_frac = 0.5

        # Query ledger: every (arch, val_acc) ever paid for.
        self.history = []
        self._seen = set()

        self.best_arch = None
        self.best_val_acc = -1.0

    def _propose(self, epoch):
        """Next architecture under the declared coverage/refinement split.

        Coverage phase: uniform random valid cells. Refinement phase:
        1-edge mutations of the current incumbent. Both phases avoid
        re-buying ledger entries when possible (repeat queries would
        still be charged by the harness).
        """
        n_explore = max(1, int(round(self.explore_frac * self.num_epochs)))
        for _ in range(20):
            if epoch < n_explore or self.best_arch is None:
                cand = random_architecture()
            else:
                cand = mutate_architecture(self.best_arch)
            if is_valid_arch(cand) and tuple(cand) not in self._seen:
                return cand
        return random_architecture()

    def _evidence_score(self, arch, val_acc):
        """Corroborated quality of a candidate (final-commitment criterion).

        Blends the candidate's own validation accuracy with the mean
        queried accuracy of its 1-edit neighbours: a cell sitting on an
        observed high plateau outranks an isolated spike with the same
        validation number. The placeholder blend is mild — redesign
        freely (surrogate consensus, rank stability, ...).
        """
        nbrs = {tuple(n) for n in get_neighbors(arch)}
        seen = [acc for a, acc in self.history if tuple(a) in nbrs]
        if not seen:
            return val_acc
        return 0.85 * val_acc + 0.15 * (sum(seen) / len(seen))

    def search_step(self, epoch):
        """Run one step of the search algorithm.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        arch = self._propose(epoch)
        val_acc = self.api.query_val_accuracy(arch)
        self.history.append((arch, val_acc))
        self._seen.add(tuple(arch))

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Commit by corroborated evidence over the ledger, not raw argmax."""
        if not self.history:
            return self.best_arch
        best, best_s = None, float("-inf")
        for arch, acc in self.history:
            s = self._evidence_score(arch, acc)
            if s > best_s:
                best, best_s = arch, s
        return best
