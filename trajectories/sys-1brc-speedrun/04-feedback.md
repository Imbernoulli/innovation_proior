Measured result — branchless 16-byte SWAR delimiter scan + SWAR number parse over raw `Unsafe` addresses,
flyweight-`byte[]` open-addressing entries, subprocess unmap trick (`baseline:swar-branchless-parser`,
`CalculateAverage_royvanrijn.java`). Metric: wall-clock m:ss.mmm, **lower is better**.

| configuration | wall-clock (m:ss.mmm) |
|---|---|
| naive streams baseline | 04:49.679 |
| parallel mmap (spullara) | 00:05.979 |
| Vector API SIMD (merykitty) | 00:03.210 |
| + SWAR-only branchless parser over Unsafe (royvanrijn) | **00:02.157** |

The published leaderboard time for this entry on the evaluation machine (eight cores, 21.0.2-graal, GraalVM
native binary) is 00:02.157 — about a 1.49× speedup over the Vector-API rung, and ~134× over the baseline.
The author's in-file changelog records the same descent on their development machine, bottoming out around
1200 ms there.
