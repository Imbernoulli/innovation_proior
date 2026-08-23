class CustomHPOStrategy:
    """Measurement-error-discipline HPO scaffold (variant re-aim).

    Sub-full-fidelity scores are treated as measurements through a
    biased, dispersed channel. The channel is characterised by
    repetition: the same configuration is submitted twice, once at
    CAL_FIDELITY and once at 1.0, and each (cheap, full) pair is one
    sample of the channel's error. The strategy maintains the running
    error estimate (gap_mean, gap_std); the incumbency rule is that
    only full-fidelity scores may claim the best — a cheap number, no
    matter how flattering, is a hypothesis about the full channel.

    Placeholder policy (weak on purpose): CAL_PAIRS calibration pairs
    are collected up front — cheap measurement, then the identical
    configuration re-measured at fidelity 1.0 — after which proposals
    are uniform random at full fidelity. The calibration estimate is
    computed and stored but not yet exploited: using it (bias-corrected
    screening, error-gated promotion, choosing the exchange rate
    between calibration spend and cheap-trial savings) is the redesign
    surface.

    Determinism note: this harness returns identical scores for an
    identical (config, fidelity) submission, so all measured error
    lives in the cross-fidelity gap — which is exactly the channel the
    pairs sample.

    Args:
        seed: random seed for reproducibility

    Returns from suggest():
        config: dict mapping param names to values
        fidelity: CAL_FIDELITY for the cheap half of a pair, else 1.0.
    """

    CAL_PAIRS = 4        # paired repeated trials invested in calibration
    CAL_FIDELITY = 0.5   # the cheap channel being characterised

    def __init__(self, seed: int = 42):
        """Initialize the strategy and the calibration bookkeeping."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pairs = []          # [(cheap_score, full_score), ...]
        self.gap_mean = 0.0      # E[full - cheap], updated per pair
        self.gap_std = 0.0       # spread of (full - cheap)
        self._cal_cfg = None     # configuration currently being paired
        self._cheap_score = None
        self._stage = "idle"     # idle -> cheap_sent -> cheap_done -> full_sent

    def _absorb(self, history: List[Trial]) -> None:
        """Fold the newest trial into the pairing state machine."""
        if not history:
            return
        last = history[-1]
        if self._stage == "cheap_sent":
            self._cheap_score = last.score
            self._stage = "cheap_done"
        elif self._stage == "full_sent":
            self.pairs.append((self._cheap_score, last.score))
            gaps = [f - c for c, f in self.pairs]
            self.gap_mean = float(np.mean(gaps))
            self.gap_std = float(np.std(gaps))
            self._cal_cfg = None
            self._cheap_score = None
            self._stage = "idle"

    def suggest(
        self,
        space: SearchSpace,
        history: List[Trial],
        budget_left: int,
    ) -> Tuple[Dict[str, Any], float]:
        """Collect calibration pairs first; then full-fidelity proposals.

        Each pair is the same configuration measured through both
        channels; everything after calibration is charged at 1.0 so the
        incumbent is always a full-fidelity fact.
        """
        self._absorb(history)

        if len(self.pairs) < self.CAL_PAIRS and budget_left > 2:
            if self._stage == "idle":
                self._cal_cfg = space.sample_uniform(self.rng)
                self._stage = "cheap_sent"
                return dict(self._cal_cfg), self.CAL_FIDELITY
            if self._stage == "cheap_done":
                self._stage = "full_sent"
                return dict(self._cal_cfg), 1.0

        config = space.sample_uniform(self.rng)
        return config, 1.0
