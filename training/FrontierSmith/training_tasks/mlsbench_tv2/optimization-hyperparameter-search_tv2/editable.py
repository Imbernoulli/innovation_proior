class CustomHPOStrategy:
    """Small-budget, full-fidelity sequential HPO scaffold (variant re-aim).

    Regime under study: the budget is a handful of expensive,
    irreversible experiments. Every trial is taken at fidelity 1.0 —
    no cheap screens, no partial runs — so each suggestion costs a
    whole unit and returns one trustworthy number. The design problem
    is information per trial: each proposal must be justified by what
    the full-fidelity history implies about where improvement is still
    plausible, and the exploration-vs-refinement decision is made anew
    with every irreplaceable unit spent.

    Placeholder policy (weak on purpose, the redesign surface):
      * a tiny declared seed batch of uniform draws (SEED_TRIALS);
      * afterwards, a local perturbation of the incumbent whose step
        scale shrinks as the career runs out — refinement tightens as
        trials become more precious. Perturbations act in transformed
        space (log axes perturbed in log units), integers re-rounded,
        and the incumbent's categorical choice is kept most of the
        time.

    One policy serves all three benchmarks (mixed float/int/categorical
    parameters, log scales, 3-6 dims): nothing is tuned per benchmark.

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: always 1.0 in this variant — a whole, trustworthy
                  experiment per suggestion.
    """

    SEED_TRIALS = 5       # declared opening batch (uniform, full fidelity)
    STEP_MIN = 0.05       # step scale (fraction of span) when budget is gone
    STEP_MAX = 0.35       # step scale when the career is still fresh
    CAT_KEEP = 0.8        # probability of keeping the incumbent's category

    def __init__(self, seed: int = 42):
        """Initialize the strategy.

        Stores the seed, creates the RNG, and remembers the first-seen
        budget so the step schedule can be expressed as a fraction of
        the whole career.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._career = None  # budget_left observed on the first call

    def _perturb(self, space: SearchSpace, base_cfg: Dict[str, Any],
                 scale: float) -> Dict[str, Any]:
        """One local move around base_cfg, respecting parameter geometry."""
        cfg = {}
        for p in space.params:
            v = base_cfg.get(p.name)
            if p.type == "categorical":
                if self.rng.rand() < self.CAT_KEEP and v in p.choices:
                    cfg[p.name] = v
                else:
                    cfg[p.name] = p.choices[self.rng.randint(len(p.choices))]
                continue
            if p.log_scale:
                lo, hi = math.log(p.low), math.log(p.high)
                x = math.log(max(float(v), 1e-300))
            else:
                lo, hi = float(p.low), float(p.high)
                x = float(v)
            x += self.rng.normal(0.0, scale * (hi - lo))
            x = min(max(x, lo), hi)
            val = math.exp(x) if p.log_scale else x
            cfg[p.name] = int(round(val)) if p.type == "int" else float(val)
        return cfg

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Propose the next configuration; fidelity is always 1.0.

        Seed batch first; then local refinement of the incumbent with a
        step scale annealed against the remaining career.
        """
        if self._career is None:
            self._career = float(max(budget_left, 1))
        if len(history) < self.SEED_TRIALS:
            return space.sample_uniform(self.rng), 1.0

        incumbent = max(history, key=lambda t: t.score)
        remaining = float(budget_left) / self._career
        scale = self.STEP_MIN + (self.STEP_MAX - self.STEP_MIN) * remaining
        config = self._perturb(space, incumbent.config, scale)
        return config, 1.0
