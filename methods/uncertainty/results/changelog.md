# Changelog

## 2026-08-20 — svfix repair pass (W3_primary_plus_ancestors)

Fixed a real math error flagged by an independent verifier of a prior fix attempt.
`reasoning.md`'s decisive step (the mixed regression+classification uncertainty-weighting
derivation) claimed that after reparameterising to the log-variance `s := log σ²`, the
classification term lands at `exp(-s)L + s` "for free," the same shape the regression term
reaches after an explicit ×2. This is false: `log σ = s/2` (an identity the trace itself
uses earlier), so the classifier's term is `exp(-s)L + s/2`, not `exp(-s)L + s`. Reaching
`+s` requires doubling the whole term, which doubles the precision coefficient too, to
`2·exp(-s)L + s` — matching the primary's own commented-out final objective
(`src/multitask.tex` L617-620: `2 exp(-s_C) L_C + exp(-s_I) L_I + exp(-s_D) L_D + s_C + s_I
+ s_D`, the explicit 2 landing only on the classification precision term).

The shipped `MultiTaskLoss` code (both classification heads: `fine_loss`, `coarse_loss` are
`cross_entropy` per `context.md`'s harness) previously implemented the uniform,
un-doubled `exp(-s_i)*L_i + s_i` for both heads — its implied fixed point `σ² = L` instead
of the derivation's true `σ² = 2L` (verified numerically both ways). Corrected the
precision coefficient to `2*exp(-s_i)` for classification heads in `reasoning.md`,
`answer.md`, and `train_answer.md` (code, formula, and all narrated numeric traces/fixed
points that depended on it). Regression-scoped passages (the `L=4`/`L=100` numerical
check, the `(1/2)exp(-s)L + s/2` sanity-check) were already correct and left unchanged.

See `notes/sources.md` for the verbatim primary quote and the independent numerical
verification of both fixed points.
