Measured result — parallel memory-mapped segments + scaled-integer parse + custom byte-keyed
open-addressing map (`baseline:parallel-mmap`, `CalculateAverage_spullara.java`). Metric: wall-clock
m:ss.mmm, **lower is better**.

| configuration | wall-clock (m:ss.mmm) |
|---|---|
| naive streams baseline | 04:49.679 |
| + parallel mmap segments, byte-keyed open-addressing map (spullara) | **00:05.979** |

The published leaderboard time for this entry on the evaluation machine (eight cores, 21.0.1-graal) is
00:05.979 — about a 48× wall-clock speedup over the baseline. The author's own in-file notes report the
same family of result on their development machine: the streams baseline at 2m37.788s and this
implementation at 0m2.013s there.
