# Mountain Meet: A Rhythm for the Single Track

## Problem
One single-track railway climbs a mountain through `S` stations `0..S-1`, joined by
`S-1` single-track **blocks** (block `b` links station `b` and `b+1`). `N` trains must
cross. Train `j` runs **eastbound** (`dir=0`, from `0` to `S-1`) or **westbound**
(`dir=1`, from `S-1` to `0`), takes `h_j` ticks to traverse **each** block (trains have
different speeds), and has a release tick `r_j`, a due tick `d_j`, a flow weight `w_j`
and a tardiness weight `v_j`.

Only some stations have a **passing loop (siding)**: station `i` can hold at most `cap[i]`
waiting trains at once. Where `cap[i]=0` a train may not stop at all (it must cross the
neighbouring blocks without pausing); the two terminals are yards with effectively
unlimited capacity. Because a single-track block holds one train at a time, two opposing
trains can only meet by having one wait in a siding while the other passes — so the whole
difficulty is choosing *where and when the meets happen*.

## Input (stdin)
```
S TMAX
cap[0] cap[1] ... cap[S-1]
N
dir r d w v h        (repeated N times)
```

## Output (stdout)
`N` lines, one per input train (input order), each with `S-1` integers
`e[0] e[1] ... e[S-2]`: the tick train `j` **enters** its 1st, 2nd, … block **in route
order** (eastbound enters physical block `0,1,…`; westbound enters physical block
`S-2,S-3,…`). Its arrival is `arr_j = e[S-2] + h_j`.

## Feasibility
Every train needs exactly `S-1` integers in `[0, TMAX]`, and:
- `e[0] >= r_j` (no early departure);
- `e[k+1] >= e[k] + h_j` (a block takes `h_j` ticks; the surplus is dwell time in the
  station reached after block `k`);
- **block exclusivity**: on each physical block, the occupancy windows `[e, e+h)` of the
  trains using it must not overlap (windows may touch). This forbids both head-on meets
  and in-block overtakes;
- **siding capacity**: at each intermediate station `i`, the number of trains *dwelling*
  there simultaneously never exceeds `cap[i]` (so where `cap[i]=0` no train may pause).

Any violation scores `Ratio: 0.0`.

## Objective (minimize)
```
F = sum_j  w_j * (arr_j - r_j)          # weighted flow time (release -> arrival)
  + sum_j  v_j * max(0, arr_j - d_j)     # weighted tardiness
```
Every train pays at least its free-run transit `(S-1)*h_j`; the score rewards shrinking
the *extra* waiting created by meets and siding conflicts, weighted by importance.

## Scoring
The checker builds an internal **serialized baseline** `B`: dispatch the trains one at a
time in release order, each departing only once the previous train has cleared the whole
line (no meets, no dwell — always feasible). With the minimization rule:
```
Ratio = min(1.0, 0.1 * B / F)
```
Reproducing the baseline scores `Ratio = 0.1`; a plan `10x` cheaper caps at `1.0`.

## Constraints
- `5 <= S <= 9`, `4 <= N <= 17`, `2 <= h_j <= 4`, `cap[i] in {0,1,2}` for intermediate
  stations. Time limit 5s, memory 512m.

## Example
`S=4`, `cap=[_,0,1,_]` (a siding only at station 2). An eastbound train `A` and a
westbound train `B`, both `h=2`, both released at `0`. Running both straight through
(`A: e = 0 2 4`, `B: e = 0 2 4`) makes them occupy the middle block `1` during the same
window `[2,4)` — a head-on, infeasible. Instead send `A` straight through (`e = 0 2 4`,
arrives `6`) and hold `B` in station 2's siding until block `1` clears, then let it
descend (`e = 0 4 6`, arrives `8`): feasible, with `2` ticks of extra waiting charged only
to `B`. The checker compares your total `F` against its serialized baseline `B`.
