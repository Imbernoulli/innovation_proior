# Changelog

## 2026-08-18 — svfix(epistemic)
- **Removed a self-supplied observation introduced by the svfix(W3_primary_only)
  pass above.** A single-turn method unit is a proposal: at that point in the
  frame the method's own experiments have not happened yet, so reasoning.md
  must not have the narrator run the hidden-state / query histogram check
  (against the primary's own App B distribution figures) and report a
  result — real numbers or not. The prior pass's rewrite of the Gaussian
  premise did exactly that: "I pull the hidden states... histogram every
  coordinate... both layers come back symmetric, single-peaked... genuinely
  Gaussian-looking," then the same reported-outcome pattern for the query
  projection (worse fit, still unimodal), the QK-normalized model (more
  pronounced mismatch, still unimodal), and the closing "checked against
  real activations" — all stated as things already observed.
- Rewrote the passage (`methods/expected-attention/results/reasoning.md:29`)
  to keep the motivation for checking rather than assuming, the
  discriminating-check DESIGN at each stage (hidden state, then query,
  then a QK-normalized model, then inside-block activations), each stage's
  PREDICTION, and an explicit decision rule ("if that check bears out... I
  build the rest of this derivation on the Gaussian query distribution; if
  it comes back multi-modal, or the tails swamp the two moments... I need a
  different distributional family before any of what follows means
  anything") — without asserting an observed outcome. The Laplacian-inside-
  a-block aside is kept as a known/prior-characterization fact (pre-dates
  this method, matches the original pre-svfix wording), not a claimed
  on-page measurement.
- Landing (Gaussian `h ~ N(mu, Sigma)` as the working premise the rest of
  the derivation is built on) is now explicitly a premise to be verified,
  not an already-verified fact — the histogram check itself is still open
  in proposal voice and belongs in a trajectory observation turn.
- The algebra that follows (linear pushforward `q = R W_Q h -> N(...)`,
  RoPE-rotation averaging, the MGF closed form) is untouched on-page
  computation and was not affected by this pass.
