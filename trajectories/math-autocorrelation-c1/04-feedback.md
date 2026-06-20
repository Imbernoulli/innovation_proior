Measured result — `construct:frontier-largeN` (endpoint: diverse-start trust-region Sequential LP at `N=600` —
rung-3 shape plus boundary-spike inits — then a long full-constraint SLP polish with restart kicks, fixed seeds).
`R` from the FFT autoconvolution evaluator on the returned `600` heights. Runtime `~20 min` total across the search
and polish phases.

| Stage | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| slp-refine (prev rung) | 600 | 1.517237 |
| diverse-start SLP search (best basin) | 600 | 1.517146 |
| + long full-constraint SLP polish (returned) | 600 | 1.517040 |

Reference points: flat ceiling `2.0`, AlphaEvolve 600-step `1.5053`, TTT-Discover 30000-step `1.5028628983`,
AutoEvolver record 30000-step `1.5028628969`, provable floor `1.28`.

Notes: `−0.000197` below the rung-3 value, reaching `R = 1.5170399450` at `N=600` — the genuine frontier this single
SLP constructor reaches. The two added levers each paid a little: the diverse multi-start search (rung-3 shape plus
left- and right-heavy boundary-spike seeds) confirmed the rung-3 basin is the lowest of those tried and nudged it to
`1.517146`, and the long full-constraint polish ground it to `1.517040`, still inching down at the time budget. A
repeat-lift to `N=1200` was tried but did not pay within budget — the per-round LP cost rises with the grid and the
finer grid did not recover below the lift value — so the returned frontier stays at `N=600`. The returned profile is
the peak-suppressing boundary-spike structure: a spike `~9.6×` the mean at index 0, `~216` of `600` heights near zero,
the middle third thinned to `~0.70×` the mean, and `~403` autoconvolution nodes within `10^-3` of the peak — the
flat-top plateau the minimax LP drives down together.

The endpoint lands `~0.0117` above the `600`-piece AlphaEvolve `1.5053` and `~0.0142` above the record
`1.5028628969`. That residual gap *at the same resolution* is the honest signature of a single bounded constructor:
a local trust-region SLP, even diversified over several basins and polished long, converges into a good basin but
not the global one AlphaEvolve found with an agentic search, and the record `1.5028628969` is a `30000`-piece
deliberately irregular construction (AutoEvolver, Claude/Opus "aspiration prompting"; after TTT-Discover's
`30000`-piece `1.5028628983` and AlphaEvolve's `600`-piece `1.5053`) found over tens of hours with two orders of
magnitude more pieces and vastly more compute. There is no finale here that reaches `1.50286`, because that requires
a large-scale agentic/evolutionary search, not a single SLP constructor — the analogue of the evolutionary-search
record standing above the single-gradient frontier in the `C2` sibling. The gap from `1.5170` down to `1.50286`, and
the further gap to the provable floor `1.28`, is the still-open part of the first autocorrelation inequality.
