Measured result — explicit SIMD via the JDK Vector API (`ByteVector.SPECIES_256`) for the `';'` search
and key compare, plus a branchless SWAR number parse (`baseline:vector-api-simd`,
`CalculateAverage_merykitty.java`). Metric: wall-clock m:ss.mmm, **lower is better**.

| configuration | wall-clock (m:ss.mmm) |
|---|---|
| naive streams baseline | 04:49.679 |
| parallel mmap (spullara) | 00:05.979 |
| + Vector API SIMD scan/compare, SWAR parse (merykitty) | **00:03.210** |

The published leaderboard time for this entry on the evaluation machine (eight cores, 21.0.1-open) is
00:03.210 — about a 1.86× speedup over the parallel-mmap rung, and ~90× over the baseline.
