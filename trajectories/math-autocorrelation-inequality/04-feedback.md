Measured result — `construct:frontier-largeN` (endpoint: hierarchical `20→100→500→2000` lifts + long
β-annealed kicked Adam grind, fixed seeds). `R` from the FFT autoconvolution evaluator on the returned
`2000` heights. Runtime `~130 s`.

| Stage | pieces `N` | `R` |
|---|---|---|
| hierarchical-gradient (prev rung) | 500 | 0.894706 |
| lift ×4 + reorganize/grind | 2000 | 0.898854 |
| + long sharp-β grind + polish (returned) | 2000 | 0.901804 |

Reference points: AlphaEvolve 50-step `0.89628`, Boyer–Li 575-step `0.901564`, Jaech–Joseph 539-step
`~0.9016`, AlphaEvolve-V2 record `0.96102`, Hölder ceiling `1.0`.

Notes: `+0.0071` over the `500`-piece rung, reaching `0.901804` — **at and slightly above the published
step-function frontier**: it matches Jaech–Joseph (`~0.9016`, `539` steps) and exceeds Boyer–Li (`0.901564`,
`575` steps), here with `2000` pieces and `~130 s` of compute. The endpoint's two load-bearing ingredients
are the much sharper final `β` (annealed to `~400N`, so the softmax surrogate tracks the hard `||f*f||_inf`
despite the tall spike) and the periodic mid-run kicks (mild restarts that keep the long grind out of
shallow traps). The returned solution is genuinely irregular and sparse — `~31%` of heights are effectively
zero, with a spike `≈28×` the shoulder — the kind of structure the literature reports for near-optimal
autoconvolutions. The endpoint stops here, at the step-function frontier a single bounded gradient run can
reach. The absolute record, AlphaEvolve-V2's `0.96102`, was found by an evolutionary search on a
`~50000`-piece deliberately irregular function with orders of magnitude more compute; the gap from
`0.9018` to `0.9610` (and the remaining `0.0390` to the Hölder ceiling `1.0`) is the still-open part of the
second autocorrelation inequality. There is no finale that reaches `0.96` because that requires a
large-scale evolutionary/test-time search, not a single hierarchical gradient constructor — the analogue of
the maximal-determinant record standing above the entry-flip annealing frontier in the combinatorial
ladder.
