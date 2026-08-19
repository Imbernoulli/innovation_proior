# Changelog

## 2026-08-18 — svfix(W3_ancestors_only)
- Audited the TRIAGE-flagged decisive step (Kronecker Fisher block approximation
  `E[āāᵀ⊗ggᵀ]≈Ā⊗G` + block-diag/tridiag inversion): genuinely derived on the page, matches
  `src/kronecker10.tex` Appendix A cumulant proof and the block-tridiagonal/precision-matrix
  argument line-for-line. Left untouched — sound as is.
- Found and fixed a separate defect in the damping section: reasoning.md stated "when I tried it
  with my approximate F̃, I couldn't find any λ that gave updates as good as exact-F methods give"
  — a self-supplied ML-experiment observation. This mirrors K-FAC's own MNIST-autoencoder ablation
  (Fig. damping_rescaling in `src/kronecker10.tex`, iteration 500) and, per Martens' PhD thesis
  (`refs/martens_thesis.pdf`, Sec. 6.6.2), was genuinely found "through trial and error" by the
  authors — i.e. a real experiment the single-turn narrator cannot have actually run.
- Rewrote the passage to derive the same conclusion (a single λ can't do both jobs) from a
  first-principles argument about what the Levenberg-Marquardt adaptation signal can and can't see,
  instead of asserting a completed empirical trial. No change to the landing (still lands on
  splitting λ from a second, Kronecker-error-compensating knob).
- No factual errors found; no change to code or to answer.md/train_answer.md (neither channel
  carried the removed claim).
