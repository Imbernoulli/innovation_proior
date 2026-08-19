# Changelog

## 2026-08-18 — svfix(W3_ancestors_only) verification pass
- Quality-gate review of the decisive step (CCE-vs-MAE gradient wall -> read both
  weights as endpoints of `f_y^{q-1}` -> integrate -> `L_q=(1-f_y^q)/q`, reasoning.md
  lines 29-54): confirmed genuinely derived (the trace works the integration itself,
  going further than the primary's own exposition) and fully backed by the primary
  source already on disk (Zhang & Sabuncu, `src/main.tex` L200-260, L371, L503-511).
  TRIAGE (class D) asked for a web hunt for a self-account grounding the interpolation
  idea and the q=0.7 default; both turn out to already be stated verbatim in the
  primary, so no external source was grafted. See notes/sources.md for the full
  gate write-up and quotes. No rewrite; no factual errors found in the decisive step
  or its surrounding derivation (cross-checked against main.tex's Appendix Lemma 1/2,
  Theorem 1/2 — bounds and formulas match exactly).
