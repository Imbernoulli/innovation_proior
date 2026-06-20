Measured result — `construct:random` (random multi-start, `4,000,000` uniform configurations,
batches of `50,000`, seed `0`). Exact min triangle area over all `165` triples; returned config
verified inside `[0,1]^2`. Runtime `~46 s`.

| Method | samples | min triangle area | fraction of record |
|---|---|---|---|
| inscribed 11-gon (prev rung) | — | 0.021456 | 0.579 |
| **random multi-start** (returned) | 4,000,000 | **0.010872** | **0.294** |

Reference: Goldberg record `1/27 = 0.037037` (fraction `1.000`).

Notes: the best of four million uniform draws reaches `0.010872`, only `0.294` of the record — and
notably *below* the structured `11`-gon baseline (`0.0215`). This is the curse of dimensionality
made concrete: configurations live in `22` dimensions, and a uniform draw almost always contains
some near-collinear trio whose sliver sets a low minimum, so even the best of millions of draws is
thin. The climb across the run was slow and saturating (`0.0102` early, `0.0109` by `2M`, then
essentially flat), confirming diminishing returns — random sampling has no mechanism to *improve* a
promising configuration, it only reports the luckiest draw, discarding every good configuration the
instant the next is scored. That wasted information is the opening for the next rung: hold onto a
good configuration and nudge one point at a time, accepting some worsening moves (simulated
annealing) to avoid the shallow traps that sink greedy hill-climbing.
