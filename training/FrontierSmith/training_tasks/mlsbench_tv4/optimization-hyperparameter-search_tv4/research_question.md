A score obtained below full fidelity is not a worse score; it is a
different measurement, taken through a channel with its own bias and
its own dispersion, and this harness offers a direct way to
characterise that channel: submit the identical configuration twice,
once cheaply and once at fidelity one, and the difference between the
two returned numbers is a sample of the cheap channel's error. This
variant requires the strategy to be built around exactly that act of
repetition.

Three duties follow from taking measurement error seriously. First,
measure the
measurement: some budget must be invested in paired trials whose only
purpose is estimating how cheap scores deviate from full ones on the
benchmark at hand — an estimate the strategy maintains and updates
rather than assumes. Second, gate influence on calibration: a
low-fidelity number may steer the search only through the lens of the
estimated error, and a configuration may claim incumbency only on a
full-fidelity measurement, never on a cheap one however flattering.
Third, spend the savings deliberately: calibration costs budget, cheap
trials save it, and the exchange rate between the two is the design's
central quantity.

The two scored quantities audit the discipline from opposite ends.
best_val_score is, by the second rule, always a full-fidelity fact,
so it reflects directly
whether calibrated screening surfaced configurations that a
full-fidelity-only search would have missed; convergence_auc registers
whether trusting the cheap channel truly accelerated the incumbent's
climb — or just burned budget rehearsing.

The claim to defend: an explicitly calibrated cheap channel beats both
extremes — the strategy that takes every cheap score at face value and
the one that refuses fidelity reduction entirely — and the run's own
paired measurements are the evidence that the calibration was real
rather than asserted.
