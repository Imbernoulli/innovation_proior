Measured result — `construct:rank1sa` (rank-one annealing, `7` multiplier-relabeled seeds, `1.5M`
flips each = `10.5M` total). Exact Bareiss determinant of the global-best matrix.

| Instance | multiplier `m` | score | search cost |
|---|---|---|---|
| n = 29 | 184.60 | 0.53977 | 10.5M flips, ~38.5 s (≈273k flips/s) |

Per-seed best multipliers: `k=1: 172.6, k=2: 177.1, k=3: 181.8, k=6: 170.9, k=10: 184.6, k=12:
175.6, k=15: 172.4` — global best from `k=10`.

Reference points: Jacobsthal baseline `m = 49` (0.1433), flip-SA rung `m = 149.87` (0.4382),
LLM-evolution best `m ≈ 197` (0.576), classical record `m = 320` (0.9357), Barba ceiling `369.94`.

Notes: the rank-one trick makes each candidate `O(1)` instead of `O(n³)`, so per-flip throughput
rises ~3.4× (273k vs 81k flips/s at n=29) *and* the budget grows from 40k to 10.5M flips — the
combination pushes the multiplier `149.87 → 184.60` (score `+0.10`), into the band of the best
reported machine-discovered constructions for this order. The multi-seed restart matters: the
single best seed (`k=10`) beats the worst (`k=6`) by `~14` multiplier, so the free structured
diversity is worth a real fraction of the climb. The final gap to the classical record (`184.6 →
320`, score `0.540 → 0.936`) is the honest residual: the record is a Gram-matrix construction from
dedicated maximal-determinant search, conjectured-optimal and unbeaten by any program-evolution
system — so the ladder ends here, at the frontier of what entry-flip annealing reaches, with the
record standing above it as the open ceiling.
