# mixtral changelog

## 2026-08-18 — epistemic correction (svfix follow-up)
- `results/reasoning.md` (top-K softmax gate paragraph, added by the prior svfix pass): that pass
  grounded the gate-collapse risk with a real ancestor failure mode (Shazeer 2017 sparse-MoE: the
  gating network self-reinforces onto a few experts) but resolved it by having the narrator claim to
  have already run training and report a per-layer top-pick distribution ("What comes back is in the
  14%-28% range... holds up on its own"). That is an own-method observation in proposal voice, which
  this single-turn unit is not entitled to (its own experiments have not happened yet).
- Fix: removed the claimed observation and the invented numbers. Kept the hypothesis (self-reinforcing
  gate collapse, sourced from the ancestor MoE construction), the discriminating-check DESIGN (track
  per-layer top-pick frequency across a spread of data domains once training is running), the
  PREDICTION (uniform ~1/8 per expert if healthy; concentration onto one or two experts if collapsing),
  and the decision rule (spread ⇒ plain gate is sufficient and the noise/balancing machinery is unneeded
  weight; concentration ⇒ add the noise term and the balancing loss). No other passage in the corpus
  repeated the removed numbers; the landing (plain top-K softmax gate, no importance-balancing loss)
  does not depend on the removed observation — it now stands on the decision rule instead.
- This unit is queued for trajectory conversion so the actual expert-utilization check can be run and
  its real outcome recorded as an observation turn, not narrator prose.
