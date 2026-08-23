class CustomHPOStrategy:
    """Cold-start-from-structure HPO scaffold (variant re-aim).

    The subject is the opening: the trials proposed before scores
    exist. They must be a function of the declared search space and
    the seed only — no defaults memorised from other problems, no
    benchmark recognition. Structure is informative: log-marked axes
    are covered in log units (uniform raw coverage of such an axis
    covers almost nothing), bounded axes are stratified rather than
    sampled blind, and every categorical choice is forced to appear
    early.

    Placeholder policy (the redesign surface): a one-shot stratified
    design of k = min(2*dim + 2, 12) configurations. Each numeric axis
    is cut into k equal quantile bands in its transformed geometry and
    the bands are assigned to trials by an independent random
    permutation (a Latin-hypercube-style opening); categorical axes
    cycle their choices under a permutation so all appear as evenly as
    possible. After the design is exhausted the scaffold falls back to
    uniform proposals — deliberately weak, so the value of the opening
    itself is what shows in the metrics.

    All evaluations run at full fidelity; the opening is about WHERE
    the first trials go, not how much they cost.

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: always 1.0 in this scaffold.
    """

    def __init__(self, seed: int = 42):
        """Initialize the strategy: seed, RNG, and an unbuilt design."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._design = None   # built lazily from the first-seen space
        self._next = 0

    def _axis_value(self, p: HParam, q: float):
        """Map a quantile q in [0, 1] to a value on axis p, respecting
        the axis's declared geometry (log-scaled axes cut in log units)."""
        if p.log_scale:
            lo, hi = math.log(p.low), math.log(p.high)
            v = math.exp(lo + q * (hi - lo))
        else:
            v = float(p.low) + q * (float(p.high) - float(p.low))
        return int(round(v)) if p.type == "int" else float(v)

    def _build_design(self, space: SearchSpace) -> List[Dict[str, Any]]:
        """Stratified opening derived from the declaration alone."""
        k = min(2 * space.dim + 2, 12)
        columns: Dict[str, list] = {}
        for p in space.params:
            if p.type == "categorical":
                idx = self.rng.permutation(k) % len(p.choices)
                columns[p.name] = [p.choices[int(i)] for i in idx]
            else:
                order = self.rng.permutation(k)
                columns[p.name] = [
                    self._axis_value(p, (int(r) + 0.5) / k) for r in order
                ]
        return [
            {name: col[i] for name, col in columns.items()} for i in range(k)
        ]

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Serve the structural opening first, then a weak uniform tail.

        The design is built once from the declaration; every design
        point is proposed at full fidelity in permuted order.
        """
        if self._design is None:
            self._design = self._build_design(space)
        if self._next < len(self._design):
            config = dict(self._design[self._next])
            self._next += 1
            return config, 1.0
        return space.sample_uniform(self.rng), 1.0
