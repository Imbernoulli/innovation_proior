# Changelog

## 2026-08-18 — svfix(W3_ancestors_only)
- Decisive step (cyclical momentum 0.95->0.85->0.95) was fake-derived: the noise-scale
  algebra (g ~ eps*N/(B*(1-m))) was used to assert with logical-necessity tone that a 3x
  noise increase "is exactly what would push the effective step past stability" -- an
  empirical claim dressed as a proof. Also removed an unsupported/incorrect attribution
  ("Sutskever and others already noted that a high constant momentum behaves like a
  pseudo-increasing learning rate") -- checked Sutskever, Martens, Dahl & Hinton 2013 and
  it describes a monotonically increasing momentum schedule for convergence speed, not the
  pseudo-increasing-LR mechanism.
- Rewrote results/reasoning.md's decisive-step passage: algebra now establishes the coupling
  exists and is large but does not settle its direction; direction is settled by the real
  discriminating test in Smith 2018 arXiv:1803.09820 Section 4.3 (Fig. 7c / Remark 5,
  already on disk, previously unused) -- three momentum treatments (constant / rising /
  falling) swept against a fixed rising LR ramp; falling momentum wins on minimum test loss,
  initial convergence speed, and stability range.
- Propagated the same fix to results/train_answer.md's parallel momentum paragraph for
  consistency (same fake-derivation language, same correction).
- No change to the landing (final method + code): momentum bounds (0.95/0.85) and the
  inverse-cosine cycling were already correct; only the justification changed.
- Source: methods/one-cycle/refs/disciplined_hyperparams_1803.09820.pdf, extracted to
  refs/disciplined_hyperparams_1803.09820.txt; quotes + provenance in notes/sources.md.

## 2026-08-18 — svfix(epistemic)
- The 2026-08-18 W3_ancestors_only pass above still crossed the line: it grounded the
  decisive momentum step in a real source (Smith 2018 Sec 4.3 Fig. 7c) but wrote it into
  reasoning.md/train_answer.md in first-person past tense as the narrator's OWN executed
  experiment ("I test it directly ... Rising momentum does buy something ... Falling
  momentum is the one that wins outright") -- a single-turn proposal has no results yet,
  sourced or not.
- Rewrote both passages to keep the two hypotheses, the discriminating-experiment DESIGN
  (three-way momentum sweep -- constant / rising / falling -- against a fixed rising LR
  ramp, matched schedule and budget), each hypothesis's PREDICTION (constant/rising: buys a
  larger tolerable peak rate at the cost of higher minimum test loss; falling: wins on
  minimum test loss, convergence speed, and stability range all at once), and the decision
  rule (whichever treatment wins the comparison is the one momentum is cycled by). Removed
  the claimed observation and the "settles it" / "wins outright" framing.
- The landing (momentum 0.95->0.85->0.95, inverse-cosine cycling) is unchanged and is now
  asserted without an in-frame observation backing the direction -- expected per the
  epistemic-fix rule; this unit needs conversion to a trajectory observation turn to supply
  the actual result.
