class NASOptimizer:
    """Purchase-gated, query-frugal NAS scaffold (variant re-aim).

    Every call to ``api.query_val_accuracy`` is a purchase from a fixed
    allotment, and the organising rule of this variant is that no
    purchase happens without a prior, FREE appraisal:

      * ``self.ledger``  — every (architecture, accuracy) ever paid for;
      * ``self._owned``  — cells already bought; re-buying costs a full
        query, so it is banned outright;
      * ``_appraise``    — the pre-purchase estimate: best paid accuracy
        among a candidate's single-edit neighbours, or None when the
        ledger holds no evidence about that neighbourhood;
      * ``_propose``     — draws candidates (incumbent mutations mixed
        with fresh cells) and rejects, WITHOUT SPENDING, any whose
        appraisal falls more than ``self.margin`` accuracy points below
        the incumbent; only appraisal-cleared or evidence-free
        candidates reach the oracle.

    The placeholder appraisal is crude on purpose — the variant asks for
    a design where the spending rule itself is the contribution: better
    free estimates, a principled clearing bar, and explicit reasoning
    about when an uninformed (exploratory) purchase is worth full price.
    The returned cell is simply the best purchase in the ledger.
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

        self.ledger = []           # receipts: (arch, val_acc) per purchase
        self._owned = set()        # cells already paid for (never re-buy)
        self.margin = 2.0          # clearing bar below incumbent, in points
        self.redraws = 12          # free rejections allowed per step

        self.best_arch = None
        self.best_val_acc = -1.0

    def _appraise(self, arch):
        """Free estimate: best paid accuracy among 1-edit neighbours."""
        nbrs = {tuple(n) for n in get_neighbors(arch)}
        vals = [acc for a, acc in self.ledger if tuple(a) in nbrs]
        return max(vals) if vals else None

    def _propose(self, epoch):
        """Draw until a candidate clears the bar; never re-buy a cell."""
        for _ in range(self.redraws):
            if self.best_arch is not None and random.random() < 0.7:
                cand = mutate_architecture(self.best_arch)
            else:
                cand = random_architecture()
            if not is_valid_arch(cand) or tuple(cand) in self._owned:
                continue
            est = self._appraise(cand)
            if est is None or est >= self.best_val_acc - self.margin:
                return cand
        # Every redraw failed the bar: fall back to an unowned fresh cell.
        for _ in range(50):
            cand = random_architecture()
            if tuple(cand) not in self._owned:
                return cand
        return random_architecture()

    def search_step(self, epoch):
        """One gated purchase per step; the metrics dict is the receipt.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        arch = self._propose(epoch)
        val_acc = self.api.query_val_accuracy(arch)
        self.ledger.append((arch, val_acc))
        self._owned.add(tuple(arch))

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Return the best purchase recorded in the ledger."""
        return self.best_arch
