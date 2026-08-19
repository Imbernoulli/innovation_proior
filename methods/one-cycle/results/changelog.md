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
