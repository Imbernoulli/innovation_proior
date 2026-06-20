Measured result — `construct:grid` (structured baselines). Exact min triangle area over all
`C(11,3) = 165` triples; all points verified inside `[0,1]^2`. Deterministic (no seed).

| Configuration | min triangle area | fraction of record |
|---|---|---|
| `4×3` grid minus one node | 0.000000 | 0.000 |
| equally-spaced boundary ring | 0.000000 | 0.000 |
| **inscribed regular 11-gon** (returned) | **0.021456** | **0.579** |

Reference: Goldberg record `Δ(11) = 1/27 = 0.037037` (fraction `1.000`).

Notes: the two lattice-like configurations score exactly `0` as predicted — both contain collinear
triples (grid rows; multiple points sharing a square edge), and one zero-area triangle is enough to
zero the minimum. The inscribed regular `11`-gon has no collinear triple (at most two points of a
circle lie on any line), so all `165` triangles are positive; its score `0.021456` is set by the
thinnest near-adjacent rim triples and lands at `0.579` of the record. This is a legal,
parameter-free floor: positive, principled, and clearly short of `1/27`. The entire remaining gap
must come from moving points to fatten the worst triangle — no closed form reproduces the irregular
record configuration. Next rung: random multi-start, the simplest search that needs no schedule.
