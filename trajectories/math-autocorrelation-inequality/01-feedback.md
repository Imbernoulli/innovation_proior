Measured result — `construct:uniform` (flat height vector), `R` from the FFT autoconvolution evaluator.
Deterministic (no seed).

| Instance (N pieces) | `R = ||f*f||_2^2 / (||f*f||_inf·||f*f||_1)` |
|---|---|
| N = 1 | 0.666667 |
| N = 10 | 0.666667 |
| N = 50 | 0.666667 |
| N = 1000 | 0.666667 |

Reference points: Matolcsi–Vinuesa 20-step `0.88922`, AlphaEvolve 50-step `0.89628`, Boyer–Li 575-step
`0.901564`, AlphaEvolve-V2 record `0.96102`, Hölder ceiling `1.0`.

Notes: the value is exactly `2/3` and is invariant under the piece count `N`, confirming the triangle
("tent") autoconvolution analysis and the dilation/refinement invariance of the functional — refining a
flat profile buys nothing. This matches the scaffold's "flat = floor" exactly, so the evaluator is trusted
for the harder rungs. The profile is rigid: every piece is identical, the autoconvolution is locked to a
triangle, and there is no internal gradient to follow. The whole gap from `0.6667` to the frontier near
`0.96` must be bought by introducing variation among the heights — which is the next rung.
