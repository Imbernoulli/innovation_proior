# Belt Duet: Scheduling the Shared Zone

## Problem
A parts belt runs along integer positions `0..W`. Two robot arms service it: arm **L** can
physically reach positions `[0, zhi]`; arm **R** can reach `[zlo, W]`. Positions inside
`[zlo, zhi]` — the **zone** — are reachable by *both* arms, but the zone is a single contested
resource: the two arms may never be working inside it at the same time.

`n` parts are numbered `0..n-1`. Part `i` sits at position `pos_i`, becomes available at time
`t_i`, and must be fully picked no later than time `e_i` (it vanishes after that). Picking part
`i`, once begun, takes exactly `pickdur_i` ticks. Each arm starts at its home position (`posL0`
for L, `posR0` for R) at time 0 and works strictly sequentially, chasing one part at a time.
Moving an arm between any two of its "nodes" (its own home, plus every part position) costs a
fixed number of ticks from an explicit, arm-specific distance table — the two arms have different
kinematics, so `L`'s and `R`'s tables differ. After arriving, the arm immediately starts picking,
or idles until `t_i` if it arrives early.

A **job** is one arm's move-plus-pick for a single part. You choose, for every job, the exact tick
`depart` at which the arm leaves its previous location (must be `>=` the tick the arm became free
from its previous job — it cannot depart early, but MAY wait idle longer if you want). A job
*traverses the zone* if the part's position lies in `[zlo, zhi]`, **or** the arm's position
immediately before the job does. Such a job occupies the zone for its *entire* duration, from
`depart` until its pick finishes. **Two zone-occupying jobs, one from each arm, must never overlap
in time** (half-open intervals; touching endpoints are fine). This is the mutex — since no other
channel lets an arm dodge it, choosing `depart` times is the only way to schedule zone access.

## Input (stdin)
```
W zlo zhi
posL0 posR0
n
pos_0 t_0 e_0 value_0 pickdur_0
...
pos_{n-1} t_{n-1} e_{n-1} value_{n-1} pickdur_{n-1}
TL[0][0] TL[0][1] ... TL[0][n+1]        (n+2 rows, arm L's move-time table)
...
TR[0][0] TR[0][1] ... TR[0][n+1]        (n+2 rows, arm R's move-time table)
```
Table node order is `[homeL, homeR, part_0, part_1, ..., part_{n-1}]`; only the rows/columns an
arm can actually reach from are ever used. All values are non-negative integers.

## Output (stdout)
Whitespace-separated tokens (newlines optional):
```
kL  i_1 d_1  i_2 d_2  ...  i_kL d_kL     (part index, chosen depart tick -- arm L, visiting order)
kR  j_1 f_1  j_2 f_2  ...  j_kR f_kR     (part index, chosen depart tick -- arm R, visiting order)
```
`0 <= kL, kR <= n`. A part index may appear in at most one of the two lists, and at most once.

## Feasibility
Replay both arms' actions in listed order; an arm becomes "free" at the tick its previous pick
finished (0 initially). The plan is feasible iff, for every listed part: its `depart` tick is
`>=` the arm's free tick, the part is reachable by that arm, its pick finishes at or before `e_i`,
and no zone-occupying job of L overlaps in time with any zone-occupying job of R. Any violation
(early depart, unreachable part, missed deadline, duplicate/out-of-range index, mutex overlap,
malformed output) scores **0**.

## Objective
Maximize `F`, the total value of every part appearing in either list (all such parts are, by
feasibility, successfully picked).

## Scoring
The checker also computes `B`, a deliberately trivial reference value: for each arm ALONE, and
never even entering the shared zone (only its own exclusive flank), the best plan obtained by
visiting reachable parts in earliest-deadline-first order, greedily keeping any that still fits;
`B` is the better of the two single-arm, zone-avoiding values. Then
```
Ratio = min(1000, 100 * F / B) / 1000
```
Matching that do-not-bother-with-the-zone reference exactly scores `0.1`; any plan that actually
uses two arms and the shared zone should clear it comfortably, with the score capped well short
of `1.0`.

## Constraints
`1 <= n <= 50`, `0 <= W <= 150`, `0 < zlo < zhi < W`, time limit 2-5s, memory 512MB.

## Example
`W=10, zlo=4, zhi=6, posL0=3, posR0=7`. Part `0=(pos=1, t=0, e=20, value=4, pickdur=1)` sits on L's
flank; part `1=(pos=5, t=0, e=5, value=9, pickdur=1)` sits in the zone. Suppose `TL[homeL][0]=2`,
`TL[0][1]=1`. Plan `kL=2: (0,0) (1,3)`, `kR=0`: L departs at 0 for part 0 (not in zone), arrives
at 2, picks until 3; departs at 3 for part 1, arrives at 4, picks until 5 (<= e=5): feasible, and
this job traverses the zone, but with no rival R job there's no mutex issue. `F=4+9=13`. Baseline
`B`: arm L alone, avoiding the zone, gets only part 0 (`4`); R can't reach either part; `B=4`.
`Ratio = min(1000, 100*13/4)/1000 = 0.325`.
