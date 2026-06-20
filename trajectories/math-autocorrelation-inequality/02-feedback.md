Measured result — `construct:coarse-anneal` (`N = 20` simulated annealing + softmax-`L_inf` L-BFGS ladder,
seed `2024`). `R` from the FFT autoconvolution evaluator on the returned heights.

| Instance | pieces `N` | `R` |
|---|---|---|
| coarse-anneal (embedded profile) | 20 | 0.884823 |
| flat baseline (prev rung) | any | 0.666667 |

Reference points: Matolcsi–Vinuesa 20-step `0.88922`, AlphaEvolve 50-step `0.89628`, Boyer–Li 575-step
`0.901564`, AlphaEvolve-V2 record `0.96102`.

Notes: `+0.2182` over the flat floor — the jump confirms that accepting downhill (worse-`R`) moves is what
clears the rigid-triangle basin a greedy climber cannot leave. The converged `20`-piece profile is
genuinely sparse and asymmetric: heights `v_0, v_1, v_4..v_7` are driven to exactly `0`, leaving a tall
spike (`v_2 = 1.0`) plus a shaped shoulder (`v_8..v_19` tapering from `~0.5` to `~0`), so the best coarse
construction uses a specific support, not all `20` pieces uniformly. The value lands `0.0044` below the
published Matolcsi–Vinuesa `0.88922` — the same band, short of it because a short bounded run on `20`
coarse pieces caps how flat the autoconvolution's cap can get. That resolution ceiling, not the search
idea, is the wall: the next rung lifts this optimized shape onto a much finer grid and refines with a
gradient method that can carve structure `20` heights cannot represent.
