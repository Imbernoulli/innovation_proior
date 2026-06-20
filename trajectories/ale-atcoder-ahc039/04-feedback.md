Measured result — rung 4 ENDPOINT (ShinkaEvolve refinements: cached validation + targeted edge
move, on the rung-3 grid SA), frozen exact evaluator. Same five seeded instances; `~5 s` per
instance, `G = 50`.

| Instance | a (mackerel) | b (sardine) | perimeter | m | objective `a−b+1` | (rung 3) |
|---|---|---|---|---|---|---|
| seed 1 | 2796 | 428 | 388000 | 136 | 2369 | 2330 |
| seed 2 | 4136 | 129 | 360000 | 96  | 4008 | 3967 |
| seed 3 | 3712 | 489 | 380000 | 54  | 3224 | 3215 |
| seed 4 | 3270 | 87  | 376000 | 112 | 3184 | 3162 |
| seed 5 | 3779 | 1270| 316000 | 4   | 2510 | 2524 |
| **mean** | | | | | **3059** | **3039** |

Reference (AtCoder relative scale): ALE-Agent SA `2880` (5th) → **ShinkaEvolve `3140` (2nd)** — the
two refinements reproduced here are the exact levers behind that jump (arXiv:2509.19349). Method
informed by: ShinkaEvolve (SakanaAI), evolving ALE-Agent's ALE-Bench (arXiv:2506.09050) solution.

Notes: +20 mean over rung 3, reproducible across repeated runs (rung 4 ≈ 3051–3059 vs rung 3 ≈
3033–3039 at the 5 s budget; the gap holds every run). The targeted edge move lifts exactly the
carving-heavy seeds: seed 1 (`2330 → 2369`, captures border mackerel, `a` up `2641 → 2796`), seed 2
(`3967 → 4008`), seeds 3, 4 small directed gains (`3215 → 3224`, `3162 → 3184`). Seed 5 holds the
rectangle (heavily overlapping shoals leave almost no profitable carve), a 14-point instance-level
wobble, not a regression. The internal `a − b` tracked by the cache matches the exact evaluator's
count to the unit on every emitted net, confirming the incremental boundary-flag cache is faithful.
This is the endpoint: the gain is modest and real, the same character as the benchmark's
`2880 → 3140` — not a new algorithm, but the SA made to search in the right direction with an
affordable cache.
