# Changelog - geglu

- 2026-08-18 `results/reasoning.md` expressivity paragraph: the claim "a single
  projection-then-pointwise unit ... can never produce a cross term x_j x_k for j≠k, because f
  acts on one scalar" was mathematically false for GELU (a smooth nonlinearity has nonzero
  curvature almost everywhere, so its Taylor expansion around any point does contain x_j x_k
  cross terms via f''(z)w_jw_k — verified by direct computation, e.g. GELU''(0) = 2φ(0) ≈ 0.798
  ≠ 0). Replaced with the true distinguishing fact: a single-projection unit depends on x only
  through one scalar direction (constant on every hyperplane orthogonal to that vector), while
  the two-projection product genuinely varies along two independent directions whenever
  W_{·i}, V_{·i} aren't parallel — the conclusion ("reaches functions the single view cannot")
  is unchanged, only the false intermediate justification was corrected. See notes/sources.md.
- 2026-08-18 checked the TRIAGE flag on the gradient-highway passage (GTU vs. linear-gated unit,
  "imported from Dauphin 2016") against `refs/dauphin2016_glu.pdf`: the trace's Eq. 2/3
  derivation is verbatim-accurate to Dauphin, Fan, Auli & Grangier 2016 §3, explicitly
  attributed in-frame, and load-bearing (not decorative) — left unchanged; see
  notes/sources.md for the verified quotes.
