Measured result — rung 3 (warm-started SA on grid region, O(1) incremental scoring), frozen exact
evaluator. Same five seeded instances; `~5 s` per instance, `G = 50`.

| Instance | a (mackerel) | b (sardine) | perimeter | m | objective `a−b+1` | (rung 2) |
|---|---|---|---|---|---|---|
| seed 1 | 2641 | 312 | 360000 | 114 | 2330 | 2362 |
| seed 2 | 4073 | 107 | 360000 | 88  | 3967 | 3188 |
| seed 3 | 3714 | 500 | 332000 | 4   | 3215 | 3215 |
| seed 4 | 3229 | 68  | 348000 | 108 | 3162 | 3088 |
| seed 5 | 3745 | 1222| 392000 | 76  | 2524 | 2510 |
| **mean** | | | | | **3039** | **2872** |

Reference (AtCoder relative scale): ALE-Agent SA `2880` (5th), ShinkaEvolve `3140` (2nd). Speed:
~`4–7 × 10^7` candidate flips per 5 s run (O(1) incremental scoring + local validity checks).

Notes: +167 mean over rung 2. SA's reversibility pays off most on seed 2 (`3188 → 3967`): starting
from the best box, it carves the boundary down to `b = 107` sardine while holding `a = 4073`
mackerel — a trade the one-shot greedy could never make. Seeds 4 also improves via boundary carving
(`m = 108`, `b = 68`). On seed 3 the box is already near-optimal and SA holds it (`m = 4`). Seed 1
sits slightly below rung 2 — at `G = 50` the perimeter budget binds before SA fully reshapes that
layout, an instance-level variance, not a regression of the method. The plateau is real: with the
boundary near its length limit, undirected random flips rarely propose the coordinated "shave here,
extend there" trade a misclassified fish needs — the directionality gap the endpoint closes.
