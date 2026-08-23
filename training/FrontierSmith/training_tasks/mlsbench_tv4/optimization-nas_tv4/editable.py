class NASOptimizer:
    """Proxy-first NAS scaffold (variant re-aim).

    Ranking work is pushed onto signals that cost nothing. An
    architecture is six op labels, so the scaffold maintains a running
    per-(edge, op) value table — sums and counts of the paid accuracy of
    every cell observed with that op on that edge. The free score of an
    unseen cell is the mean of its six table entries (backed off to the
    global mean where a pair is unobserved). The oracle is demoted to an
    auditor:

      * a short seeding prefix (``self.seed_steps``) buys unguided cells
        so the tables hold evidence at all;
      * afterwards each step assembles a large FREE candidate pool —
        mutations of the top paid cells plus fresh random cells —
        scores the whole pool with the table, and pays for the single
        argmax (``self.pool_size`` free scores per paid query).

    Hard boundary: the harness's parameter budget check rules out
    heavyweight learned predictors, so the proxy must stay cheap
    arithmetic over structure (tables, path-encoding linear fits at
    most — ``path_encoding`` is available). The invited work is better
    free signal — pairwise edge interactions, path-level tables,
    recency weighting — never more oracle calls. The returned cell is
    the best AUDITED purchase: proxies rank, only queries decide.
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

        self.seed_steps = 8              # unguided table-seeding queries
        self.pool_size = 120             # free candidates scored per step
        self._sum = np.zeros((NUM_EDGES, NUM_OPS))
        self._cnt = np.zeros((NUM_EDGES, NUM_OPS))
        self.evidence = []               # (arch, paid accuracy)
        self._paid = set()

        self.best_arch = None
        self.best_val_acc = -1.0

    def _absorb(self, arch, acc):
        """Fold a paid observation into the per-(edge, op) tables."""
        for e, o in enumerate(arch):
            self._sum[e, o] += acc
            self._cnt[e, o] += 1.0

    def _proxy_score(self, arch):
        """Free structural estimate from the tables (global-mean backoff)."""
        total = float(self._cnt.sum())
        gmean = float(self._sum.sum()) / total if total > 0 else 0.0
        vals = []
        for e, o in enumerate(arch):
            c = self._cnt[e, o]
            vals.append(float(self._sum[e, o]) / c if c > 0 else gmean)
        return sum(vals) / len(vals)

    def _candidate_pool(self):
        """Assemble free candidates: elite mutations plus fresh cells."""
        elites = [a for a, _ in sorted(self.evidence, key=lambda t: -t[1])[:3]]
        pool, seen, attempts = [], set(), 0
        while len(pool) < self.pool_size and attempts < 20 * self.pool_size:
            attempts += 1
            if elites and random.random() < 0.5:
                cand = mutate_architecture(random.choice(elites))
            else:
                cand = random_architecture()
            key = tuple(cand)
            if key in seen or key in self._paid or not is_valid_arch(cand):
                continue
            seen.add(key)
            pool.append(cand)
        return pool if pool else [random_architecture()]

    def search_step(self, epoch):
        """Seed the tables early; afterwards pay only for the proxy argmax.

        Args:
            epoch: Current search iteration (0-indexed)

        Returns:
            dict: Metrics to log, must include 'best_val_acc' and 'queries'.
        """
        if epoch < self.seed_steps or not self.evidence:
            arch = random_architecture()
            for _ in range(10):
                if tuple(arch) not in self._paid:
                    break
                arch = random_architecture()
        else:
            arch = max(self._candidate_pool(), key=self._proxy_score)

        val_acc = self.api.query_val_accuracy(arch)
        self._paid.add(tuple(arch))
        self.evidence.append((arch, val_acc))
        self._absorb(arch, val_acc)

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_arch = arch

        return {
            "best_val_acc": self.best_val_acc,
            "queries": self.api.query_count,
            "current_val_acc": val_acc,
        }

    def get_best_architecture(self):
        """Best audited cell — the proxy proposes, only paid queries decide."""
        return self.best_arch
