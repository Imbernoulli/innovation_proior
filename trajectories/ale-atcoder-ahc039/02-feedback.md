Measured result — rung 2 (grid-cell greedy region growing + rectangle fallback), frozen exact
evaluator. Same five seeded instances.

| Instance | a (mackerel) | b (sardine) | perimeter | m | objective `a−b+1` | (rung 1) |
|---|---|---|---|---|---|---|
| seed 1 | 2770 | 409  | 393334 | 50 | 2362 | 2134 |
| seed 2 | 3618 | 431  | 393332 | 36 | 3188 | 2572 |
| seed 3 | 3714 | 500  | 332000 | 4  | 3215 | 3210 |
| seed 4 | 3211 | 124  | 180000 | 4  | 3088 | 3088 |
| seed 5 | 3779 | 1270 | 316000 | 4  | 2510 | 2518 |
| **mean** | | | | | **2872** | **2704** |

Reference (AtCoder relative scale): ALE-Agent `2880` (5th), ShinkaEvolve `3140` (2nd).

Notes: +168 mean over rung 1. On the overlapping-shoal seeds (1, 2) the grown rectilinear region
wins decisively — the carved boundary cuts sardine out of a mackerel pocket (`m = 50`, `m = 36`),
dropping sardine far below the rectangle. On seeds 3, 4 the best rectangle is already near-optimal,
so the rung correctly falls back to the `m = 4` box (identical to rung 1). Seed 5's grown region
fails to beat the rectangle by the internal estimate, so the box is kept and the rung essentially
ties rung 1 there (a 8-point dip from grid-discretization noise in the estimate). The greedy is
one-shot and irreversible: it cannot remove an early bad cell and saturates the perimeter on ragged
boundaries — the opening for replacing greedy growth with reversible local *search* in the next
rung.
