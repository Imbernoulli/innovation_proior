Measured result — `construct:coarse-slsqp` (annealed soft-max SLSQP, `12` restarts, seed `0`),
AlphaEvolve App. B.5 evaluator. `C` is the true hard-max overlap of the best returned vector.

| Stage | pieces `n` | `Σ v` (target `n/2`) | `C` (upper bound) |
|---|---|---|---|
| flat floor (prev rung) | any | — | 0.500000 |
| coarse SLSQP, best of 12 starts | 24 | 12.0000 | 0.381240 |

Reference points: White lower bound `0.379005`, AutoEvolver record `0.38086945`, AlphaEvolve `0.380924`,
Haugland `0.380927`, Erdős floor `0.5`.

Notes: `−0.1188` off the flat floor in a single rung, reaching `0.381240` at only `24` pieces — already
within `~3×10^{-4}` of the Haugland-2016 landmark `0.380927` and the AlphaEvolve `0.380924`, and within
`~4×10^{-4}` of the AutoEvolver record `0.38086945`. The returned profile is exactly feasible
(`Σ v = 12 = n/2`) and genuinely near-binary: `~29%` of the cells are pinned at `0` or `1` (the optimizer
drives heights to the box corners), with the interior cells taking intermediate values to satisfy the sum
constraint — the asymmetric, spiky structure the reasoning predicted, breaking the flat profile's
self-alignment. The two load-bearing ingredients are the soft-max surrogate (so SLSQP gets a usable
gradient through the minimax's kinks) and multi-start (the landscape is non-convex; one start lands in a
worse basin). The cap here is resolution: `24` wide cells cannot resolve the fine structure of the optimal
profile, so the worst overlap cannot be shaved below `~0.3812`. The next rung lifts this optimized profile
to many more pieces and refines it with basin-hopping, where the extra degrees of freedom carve the finer
structure that brings the bound toward the published step-function frontier.
