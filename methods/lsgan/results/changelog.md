# Changelog

## 2026-08-18 — svfix(W3_ancestors_only) numeric correction
- `results/reasoning.md`: the discretized-toy numeric self-check of the Pearson
  chi^2 identity (`p_d = N(0,1)`, several `p_g`) had one wrong row. For
  `p_g = N(0, 2)` the trace claimed both the `∫N²`-form and the `χ²`-form
  evaluate to `0.154243`; independently recomputing the same integral
  (fine-grid trapezoid, n≈1.2e7 points, and adaptive quadrature, both over a
  wide enough range to avoid tail truncation) gives `0.159982` for both
  forms, matching each other exactly (as the algebra requires) but not the
  originally written number. The other three rows (`p_g=N(0,1)`,
  `p_g=N(1,1)`, `p_g=N(-2,0.7)`) were independently reproduced to 5-6
  decimals, confirming those were genuine computations and this one row was
  a one-off arithmetic slip. Corrected `0.154243 -> 0.159982` in both
  columns; no other file referenced the wrong value.
