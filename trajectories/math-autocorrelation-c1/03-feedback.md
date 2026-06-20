Measured result — `construct:slp-refine` (Adam-softmax warm start at `N=600`, then full-constraint trust-region
Sequential LP with restart kicks, fixed seeds). `R` from the FFT autoconvolution evaluator on the returned `600`
heights. Runtime `~10 min`.

| Stage | pieces `N` | `R` (upper bound on `C1`) |
|---|---|---|
| coarse-anneal (prev rung) | 50 | 1.537084 |
| Adam-softmax warm start | 600 | 1.526568 |
| + trust-region Sequential LP grind (returned) | 600 | 1.517237 |

Reference points: flat ceiling `2.0`, AlphaEvolve 600-step `1.5053`, TTT-Discover `1.5028628983`, AutoEvolver
record `1.5028628969`, provable floor `1.28`.

Notes: `−0.0198` below the coarse rung, reaching `1.517237` at `N=600`. The two load-bearing changes are the *minimax*
reframing — the SLP lowers the whole near-tight plateau of autoconvolution nodes at once, where the softmax/subgradient
warm start plateaued near `1.5266` because it could only press down one peak node at a time — and the trust-region
discipline that keeps the quadratic linearization valid (accept only if true `R` drops; grow on success, shrink and
restart-kick on rejection). The returned profile is the expected peak-suppressing structure: a tall spike at the
boundary (`~9×` the mean at index 0), mass heavier toward both ends, the middle third thinned to `~0.70×` the mean,
and `~215` of `600` heights driven near zero. The rung lands `~0.012` above the `600`-piece AlphaEvolve `1.5053` and
`~0.014` above the record `1.5028628969`. The remaining gap to AlphaEvolve at the *same* resolution is the honest
signature of a single bounded constructor: a local trust-region SLP from one warm start converges into a good basin
but not the global one AlphaEvolve found with an agentic search and far more compute. That gap — and the room a finer
grid would open — is the opening for the endpoint: lift to more pieces and grind the SLP longer toward the record.
