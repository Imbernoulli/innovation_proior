# Changelog

- 2026-08-18 `methods/gelu/results/reasoning.md`: epistemic fix. A prior svfix pass had the narrator
  claim to actually train the raw stochastic mask (no expectation taken) as the sole nonlinearity and
  report its own results on MNIST/TIMIT/Twitter POS ("wins outright... 2.10% vs 2.00%", "tie at 29.46%",
  "loses, 12.5% against ... 11.9%") — an own-method observation a single-turn proposal is not entitled
  to (the method's own results belong only in trajectory observation turns). Rewrote in place: kept the
  discriminating-experiment design (raw sampled mask, no separate nonlinearity, matched budget, vs.
  ReLU-with-dropout, across an image/speech/tagging task), kept the hypothesis and the specific
  prediction (mask should lose wherever a task wants near-zero extra regularization, because its
  strength is welded to `Φ(x)` and not a settable knob), and kept the decision rule (switch to the
  expectation specifically because it strips out the un-tunable regularization by removing the sampling
  itself). Removed only the claimed outcomes and the numbers. No change to the landing (GELU =
  `x·Φ(x)`) or to any other file.
