Measured result — `record-construction` (load the released AlphaEvolve-V2 `~50000`-piece irregular step
function from `record_heights.json` and score it under this trajectory's own FFT autoconvolution
evaluator). No optimization; deterministic; runs in `<1 s`. The evaluator is first self-validated on
AlphaEvolve's published `50`-step function.

| Stage | pieces `N` | `R` |
|---|---|---|
| frontier-largeN (prev rung, gradient endpoint) | 2000 | 0.901804 |
| evaluator self-check (AlphaEvolve 50-step) | 50 | 0.896280 |
| **AlphaEvolve-V2 record (loaded + verified)** | **50000** | **0.961021** |

Reference points: AlphaEvolve `50`-step `0.89628`, Boyer–Li `575`-step `0.901564`, AlphaEvolve-V2 record
`0.96102`, ImprovEvolve `0.96258`, Hölder ceiling `1.0`.

Notes: `+0.0592` over the gradient endpoint, reaching `0.961021` — the published record, **verified end to
end under this trajectory's exact evaluator**, not approximated by an optimizer. The evaluator first
reproduces AlphaEvolve's original `50`-step `0.89628` to the published digits, confirming the scoring
convention matches DeepMind's; the same `autoconv_ratio` then returns `0.961021` on the loaded `~50000`-piece
record heights, matching the published AlphaEvolve-V2 value (arXiv:2511.02864). The record is *obtained*,
not reproduced from scratch: it is a deliberately irregular `~50000`-step function found by a large-scale
evolutionary / test-time search, a different shape family from the smooth spike-and-shoulder basin the
gradient endpoint (`0.9018`) saturates in — `0.90` was a shape limit, not a resolution limit, and the
`~0.059` jump is bought by the irregular construction the search produced, not by more gradient steps.
Together AI's publicly reproducible `100000`-point construction scores `0.961206` (the best with released
data); ImprovEvolve reports `0.96258` but does not release its solution, so the canonical record verified
here is AlphaEvolve-V2's `0.961021`. The residual `0.0390` to the Hölder ceiling `1.0` is the genuinely
open part of the second autocorrelation inequality — no construction, evolutionary or otherwise, has closed
it. This rung is the analogue of putting the maximal-determinant record on the combinatorial ladder above
the entry-flip annealing frontier: the record stands above the local-constructor endpoint because it lives
in a region local descent cannot reach.
