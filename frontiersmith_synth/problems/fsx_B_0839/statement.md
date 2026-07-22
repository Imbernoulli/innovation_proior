# The Alchemist's Lossy Decanting Puzzle

## Problem
An alchemist has `N` tanks, numbered `0..N-1`, each with a fixed maximum
capacity. Tank `0` starts holding `V0` units of raw tincture; every other
tank starts empty. There are `M` **valves**. Valve `e` is a directed pipe
from some tank `u` to some tank `v`, tagged with a fixed positive rational
factor `num/den` (a reduced fraction). Opening valve `e` is only possible
when the *entire* current batch of tincture sits in tank `u`: it drains tank
`u` completely, multiplies the drained volume by exactly `num/den`, and
deposits the result into tank `v` (which must not exceed its capacity).
Tank `u` is left empty; the batch now lives entirely in tank `v`. At every
moment exactly one tank holds the (nonzero) batch — you can only use valves
that start at that tank.

A **plan** is a sequence of valve operations `e_1, e_2, ..., e_R`, executed
in order starting with the batch in tank `0`. The plan is **feasible** iff
every operation is legal (correct source tank, capacity respected) and,
after the last operation, the batch sits in a designated target tank `T`
with volume inside a closed window `[Lo, Hi]`. Your task: find a feasible
plan of **minimum length `R`**.

Nothing about the valve factors' internal structure is hinted beyond "a
fixed positive rational" — discovering what makes some plans dramatically
shorter than others is the point.

## Input (stdin)
```
N M
cap_0 cap_1 ... cap_{N-1}
V0 T Nb
Lo_num Lo_den Hi_num Hi_den
b_0 b_1 ... b_{Nb-1}
u_0 v_0 num_0 den_0
...
u_{M-1} v_{M-1} num_{M-1} den_{M-1}
```
`b_0..b_{Nb-1}` is a list of `Nb` valve ids that is GUARANTEED to be a valid
plan reaching the target window if applied in that order starting from tank
`0` (a "long way round" that always works — but need not be anywhere near
shortest; valve ids carry no other positional meaning). `Lo/Hi` are exact
rationals given as numerator/denominator pairs. `1 <= N,M <= 40`,
`1 <= caps[i] <= 10^18`, `1 <= V0 <= 10^18`.

## Output (stdout)
```
R
e_1 e_2 ... e_R
```
`R` valve ids (each in `0..M-1`), whitespace-separated, `1 <= R <= 500`.

## Feasibility
Replaying the plan from `(tank 0, volume V0)` in exact rational arithmetic
must never violate a source-tank mismatch or a capacity bound, and must end
with the batch in tank `T` with volume in `[Lo, Hi]` (inclusive, exact
comparison — no tolerance is needed since all arithmetic is exact rational).
Any parse error, out-of-range valve id, `R` outside `[1,500]`, or violation
above scores `Ratio: 0.0`.

## Scoring
Let `B = Nb` (the length of the guaranteed `b_0..b_{Nb-1}` plan; the
checker itself replays it to confirm it is valid). With your plan length
`R`:
```
Ratio = min(1, 0.1 * B / R)
```
Matching the guaranteed long plan scores `0.1`. Halving your plan length
doubles the ratio; reaching a tenth of `B` caps the ratio at `1.0`. The
true shortest plan is not guaranteed to be discoverable by any efficient
general method, so real headroom remains above what straightforward search
achieves.

## Constraints
- `1 <= N,M <= 40`; time limit 5s; `2 <= Nb <= 20`.
- All arithmetic is exact (Python `fractions.Fraction` or equivalent);
  never estimate with floating point.
- Deterministic scoring; no timing or randomness anywhere in the checker.

## Example
Suppose tank 0 starts with `V0 = 1`, valve `0` goes `0 -> 1` with factor
`3/2`, and the target window around tank `1` is `[1.49, 1.51]`. The plan
`R=1`, valve list `[0]` drains tank 0 (volume 1), deposits `1 * 3/2 = 1.5`
into tank 1, which lies in `[1.49, 1.51]` — feasible, `R=1`. If the
guaranteed baseline plan for this toy instance also had length `1`, this
submission would score `min(1, 0.1*1/1) = 0.1`; a genuinely shorter plan
than the baseline scores higher, up to the `1.0` cap.
