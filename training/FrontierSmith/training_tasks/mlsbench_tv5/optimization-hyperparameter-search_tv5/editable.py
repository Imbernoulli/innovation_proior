class CustomHPOStrategy:
    """Model-committed proposer scaffold (variant re-aim): no random
    fallback.

    After a declared seed batch (dim + 3 uniform draws — the only blind
    proposals of the run), every suggestion is sampled from an explicit
    model of the history: an estimation-of-distribution proposer fitted
    to the elite quartile of past trials. There is no epsilon-greedy
    branch and no uniform fallback anywhere after seeding; exploration
    must come from the model, so the scaffold builds in two anti-collapse
    devices that the redesign should improve rather than remove:

      * a dispersion floor — each numeric axis's sampling std is never
        allowed below STD_FLOOR of its transformed span, so proposals
        cannot degenerate to photocopies of the incumbent;
      * smoothed categorical frequencies — every choice keeps non-zero
        probability (add-SMOOTH counts over the elites), so no category
        is ever permanently abandoned by the model.

    Numeric axes are modelled in their declared geometry (log-marked
    axes fitted and sampled in log units, integers re-rounded), and one
    policy serves every benchmark unchanged.

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: always 1.0 in this scaffold — the variant is about
                  where proposals come from, not what they cost.
    """

    STD_FLOOR = 0.10   # min sampling std, as a fraction of transformed span
    SMOOTH = 1.0       # additive smoothing for categorical counts

    def __init__(self, seed: int = 42):
        """Initialize the strategy: seed, RNG, seed-batch size unset."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._seed_n = None  # dim + 3, fixed on the first call

    def _model_sample(self, space: SearchSpace,
                      history: List[Trial]) -> Dict[str, Any]:
        """Sample one configuration from the elite-fitted model."""
        elite_n = max(3, len(history) // 4)
        elites = sorted(history, key=lambda t: t.score,
                        reverse=True)[:elite_n]
        config: Dict[str, Any] = {}
        for p in space.params:
            if p.type == "categorical":
                counts = np.array(
                    [self.SMOOTH + sum(1 for t in elites
                                       if t.config.get(p.name) == c)
                     for c in p.choices],
                    dtype=float,
                )
                idx = int(self.rng.choice(len(p.choices),
                                          p=counts / counts.sum()))
                config[p.name] = p.choices[idx]
                continue
            if p.log_scale:
                lo, hi = math.log(p.low), math.log(p.high)
                xs = [math.log(max(float(t.config[p.name]), 1e-300))
                      for t in elites]
            else:
                lo, hi = float(p.low), float(p.high)
                xs = [float(t.config[p.name]) for t in elites]
            mu = float(np.mean(xs))
            sd = max(float(np.std(xs)), self.STD_FLOOR * (hi - lo))
            x = min(max(mu + sd * float(self.rng.randn()), lo), hi)
            val = math.exp(x) if p.log_scale else x
            config[p.name] = int(round(val)) if p.type == "int" else float(val)
        return config

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Declared seed batch, then model-only proposals forever after.

        The seed batch size is fixed on the first call (dim + 3) and is
        never reopened; once it is spent, every configuration comes from
        _model_sample and nothing else.
        """
        if self._seed_n is None:
            self._seed_n = space.dim + 3
        if len(history) < self._seed_n:
            return space.sample_uniform(self.rng), 1.0
        return self._model_sample(space, history), 1.0
