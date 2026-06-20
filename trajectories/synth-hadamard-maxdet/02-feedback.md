Measured result — `construct:flipsa` (annealed single-entry flips from the Jacobsthal seed), RNG
seed 11, `40k` flips. Exact Bareiss determinant of the best matrix.

| Instance | multiplier `m` | score | search cost |
|---|---|---|---|
| n = 29 | 149.87 | 0.43821 | 40k flips, ~0.5 s (≈81k flips/s) |

Reference points: Jacobsthal baseline `m = 49` (0.1433), classical record `m = 320` (0.9357),
LLM-evolution `m ≈ 197` (0.576), Barba ceiling `m = 369.94`.

Notes: annealing clears the strict local maximum immediately and climbs `49 → 149.87`, a `3.1×`
jump in determinant and `+0.295` in score — confirming the diagnosis that the baseline's symmetry,
not the move set, was the wall. The multiplier is no longer an integer: `7^12` divides the
determinant only for specially structured matrices (like the record), so a searched, symmetry-broken
configuration has `m` fractional, which is expected and fine — the score normalization is defined on
`|det|` directly. The run plateaus in the low-150s of multiplier: with each candidate costing a full
`slogdet` factorization, `40k` flips from a single seed is about what is affordable, and the
remaining climb to the frontier needs far more flips and multiple restarts than full-recompute
scoring can pay for. Per-flip cost is the binding constraint — the next rung attacks exactly that.
