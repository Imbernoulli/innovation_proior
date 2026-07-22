# Standby by Design: Rostering for a Dumb Dispatcher

You publish a month-long nurse roster. When staff call out sick, a **fixed, dumb dispatcher**
(a published greedy rule you cannot change) scrambles to re-cover the open shifts. Your roster
is judged by how badly the *worst* callout scenario hurts **after** that dispatcher does its best.
Good slack is worthless unless the dispatcher can actually reach it.

## Input (stdin)
- Line: `S D C` — nurses `0..S-1`, days `0..D-1`, skill classes `0..C-1`.
- Line: `maxshift[0..S-1]` — a nurse works at most this many shifts in the month.
- Line: `base[0..S-1]` — shifts a nurse works **beyond** `base` while being called in as recovery
  cost overtime.
- `S` lines: `cnt s1 s2 ...` — the skills each nurse holds.
- For each day `d`: a line `T_d`, then `T_d` lines `skill hours` — the shift-slots that day
  (in canonical slot order).
- Line `K`, then `K` lines, each `A i1 d1 i2 d2 ... iA dA` — scenario `s` is a set of
  `(nurse, day)` callouts (that nurse is absent that day).
- Line: `U OT` — cost weights (uncovered-hour weight `U`, overtime-hour weight `OT`).

## Output (stdout)
For **every** shift-slot, in day order then slot order, print the nurse assigned to it
(whitespace-separated, exactly `sum(T_d)` integers).

## Feasibility (any violation ⇒ score 0)
- Each slot is assigned one nurse who **holds the required skill**.
- No nurse is assigned two slots on the **same day**.
- No nurse's total assignments exceed `maxshift`.
(Every slot must be covered in the base roster — there are no absences yet.)

## The fixed dispatcher (recovery simulation, per scenario)
Apply the roster; the scenario's absent nurses vanish from their day's assignments, leaving
**holes**. The dispatcher then repairs, deterministically:

> Process days `0,1,…,D-1`. On each day, take its holes in slot order. For a hole needing
> skill `k` on day `d`, scan nurses `j = 0,1,…,S-1` and take the **first** `j` that (a) holds
> skill `k`, (b) is not absent on day `d`, (c) is not already working day `d`, and (d) has not
> reached `maxshift`. Assign `j`. If that shift pushes `j`'s running total above `base[j]`, the
> shift's hours are **overtime**. If no such `j` exists, the hole stays **uncovered**.

Scenario cost `= U · (uncovered hours) + OT · (overtime hours incurred during repair)`.

## Objective (minimize)
Your score is driven by the **worst** scenario:
`cost = max over scenarios of ( scenario cost )`. Lower is better.

## Scoring
The checker builds its own baseline roster `B` (fill each slot with the lowest-index qualified
nurse) and computes its worst-scenario cost. With your worst-scenario cost `F`:
`Ratio = min(1000, 100 · B / max(1,F)) / 1000` (reproducing the baseline ⇒ `0.1`; driving the
worst scenario 10× lower caps at `1.0`). NaN/inf or infeasible output ⇒ `0`.

## Why this is hard
The dispatcher scans nurses **low-index first, one day at a time**. A reserve is only useful if
it sits at a `(nurse, day)` the scan will actually reach for a real hole. The min-load or
lowest-index roster keeps the versatile low-index specialists busy — so when a scarce-skill
nurse calls out, the only other qualified nurse is already working and the shift collapses. The
insight is to place your scarce-skill slack **exactly at the states the frozen repair heuristic
visits**, designing your roster around the dispatcher's weakness. There is no closed-form
optimum: it is a two-stage robust design against a fixed, non-convex recourse rule.

## Constraints
`S ≤ 14`, `D ≤ 8`, `K ≤ 11`. Time limit 5 s, memory 512 MB.

## Example (worked score, small)
Suppose one scarce-skill nurse (id 0) covers the only skill-2 shift every day and nurse 1 is the
sole other skill-2 holder. A scenario calls out nurse 0 on day 3. If your roster left nurse 1
working a common shift on day 3, the dispatcher finds no skill-2 reserve → that shift is
uncovered → large `cost`. If instead you covered day 3's common shifts with high-index nurses and
kept nurse 1 free, the dispatcher reaches nurse 1 and re-covers the shift (perhaps as overtime),
slashing the worst-scenario cost and raising `Ratio`.
