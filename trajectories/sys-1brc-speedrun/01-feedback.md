Measured result — the reference single-threaded streams baseline (`baseline:naive-streams-baseline`,
`CalculateAverage_baseline.java`). Metric: wall-clock m:ss.mmm, **lower is better**.

| configuration | wall-clock (m:ss.mmm) |
|---|---|
| naive streams baseline (`Files.lines` + `split` + `groupingBy`/`TreeMap`, single thread) | **04:49.679** |

This is the published reference number on the evaluation machine (eight cores of the AMD EPYC 7502P, the
challenge's own ~2-minute-class baseline as reported in the repo, here measured at 04:49.679). It is the
budget every faster rung is measured against.
