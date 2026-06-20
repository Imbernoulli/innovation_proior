Measured result — rung 1 (best axis-aligned rectangle), frozen exact evaluator (integer
point-in-rectilinear-polygon, boundary ⇒ inside). Five seeded instances, `5000` mackerel + `5000`
sardine each.

| Instance | a (mackerel) | b (sardine) | perimeter | m | objective `a−b+1` |
|---|---|---|---|---|---|
| seed 1 | 2643 | 510  | 203334 | 4 | 2134 |
| seed 2 | 4366 | 1795 | 333334 | 4 | 2572 |
| seed 3 | 3744 | 535  | 333334 | 4 | 3210 |
| seed 4 | 3201 | 114  | 180000 | 4 | 3088 |
| seed 5 | 3724 | 1207 | 313334 | 4 | 2518 |
| **mean** | | | | | **2704** |

Reference (AtCoder relative scale, not our raw objective): ALE-Agent SA `2880` (5th), ShinkaEvolve
`3140` (2nd).

Notes: every output is a legal 4-vertex rectangle, well within the perimeter budget on all seeds.
The rectangle is strong where one mackerel cluster dominates (seed 3: `a−b = 3209`) but bleeds
sardine where shoals overlap (seed 2: `b = 1795`; seed 5: `b = 1207`) — the box cannot dodge
sardine threaded through a mackerel pocket. No box can both reach outlying mackerel and exclude
interior sardine, which is exactly the convex-single-body limitation the next rung must break.
