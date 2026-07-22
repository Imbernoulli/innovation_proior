# Wavefront Filling: Self-Reconfiguring Modular Swarms

## Story
`N` identical unit modules of a robot swarm sit on a walled 2D lattice in a START
configuration and must reshape into a TARGET configuration of `N` cells over `T`
**synchronous** discrete rounds. Every round, *every* module looks at its own
4-neighbourhood (`N=(-1,0)`, `E=(0,+1)`, `S=(+1,0)`, `W=(0,-1)`) and picks ONE action —
`STAY` or move one step — from a single **local rule table**: one shared table applied to
every module and every round of *that run* (you naturally build it from the public
instance you are given, like any answer in this format — but once fixed, it drives the
whole simulation with no module ID, no global controller, and no per-round replanning).

**Moves resolve in parallel.** A move into a cell occupied at the *start* of the round is
blocked. If two or more modules request the same currently-empty cell, the request from
the module at the numerically smallest `(row, col)` wins (a fixed rule of the environment,
not extra information for any submission); the others stay and re-evaluate next round.

## Two broadcast fields
The evaluator precomputes once per instance, over every free cell, two BFS-distance
fields (walls respected): `A(cell)` = distance to the nearest target cell, `G(cell)` =
distance to the nearest seed cell (`seeds`, a public list of cells, almost always at the
target shape's far end — the family's broadcast gradient). Both are static, independent
of module positions.

## The local pattern (your table's key)
For each of its 4 neighbours (order N,E,S,W) a module computes one digit 0-8:
`0` = wall/off-grid (never movable); `1..4` = free-of-walls but occupied by another
module this round (`1`=neither field improves by moving there, `2`=only `G` improves,
`3`=only `A` improves, `4`=both — digit `= 1 + 2*impA + impG`); `5..8` = empty and
movable, same sub-coding (digit `= 5 + 2*impA + impG`). "Improves" = strictly lower at
the neighbour than at your own cell. The 4 digits in N,E,S,W order form a 4-character key
(e.g. `"0758"`).

## Candidate contract (isolated stdin -> stdout program)
`stdin`: one JSON public instance:
```json
{"name":str,"H":int,"W":int,"N":int,"T":int,
 "walls":[[r,c]...], "start":[[r,c]... N cells], "target":[[r,c]... N cells],
 "seeds":[[r,c]...]}
```
`stdout`: one JSON object: `{"table": {"XXXX": "ACT", ...}, "default": "STAY"}`. Keys are
exactly 4 characters, each `'0'`-`'8'`; `ACT` in `{"STAY","N","E","S","W"}`. An omitted key
uses `default` (or `"STAY"` if `default` is also omitted). The key alphabet is universal,
so a good table need not even inspect the instance's geometry.

### Validity
`table` has at most 7000 entries; each key is exactly 4 chars from `'0'`-`'8'`; each value,
and `default` if present, is a valid action. Any violation, crash, timeout, or non-JSON
output scores **0.0** on that instance.

## Scoring (deterministic)
The evaluator simulates your table for `T` rounds from the start configuration.
```
base = |start ∩ target| / N                 # "do nothing" overlap
obj  = |final ∩ target| / N                 # your achieved overlap
r = clamp( 0.1 + 0.9 * (obj - base) / max(1.15 - base, eps), 0, 1 )
```
Doing nothing scores exactly `0.1`. The `1.15` ceiling means even a *perfect*
reconfiguration (`obj == 1`) does not saturate `r` at `1.0` — headroom stays open above
every reference solution. Final score = mean of `r` over **10** fixed seeded instances:
independent-lane open reconfigurations, single-file "tube" traps of varying length and
shape, and held-out harder cases (a bent tube, and a swarm split across two tubes).

## Why it is open-ended
Reading only `A` and stepping toward it is the obvious first idea, and it works fine when
one module's approach never interacts with another's. It fails badly once the target
region is more than one cell deep along the approach direction (a corridor, or the
interior of a solid block): the instant a module lands on *any* target cell, its own `A`
is already the global minimum `0`, so no neighbour can ever look better under "move only
if `A` improves" — it freezes **forever**, permanently plugging the only route in and
stranding everyone behind it. No tuning of that rule fixes this; it needs a different
signal. Reading `G` *only after* `A` stops helping lets a module keep advancing deeper
while unclaimed, closer-to-seed cells exist, so the first arrival walks to the true dead
end and later arrivals settle progressively closer to the entrance — a self-organizing
conveyor filling far-to-near, built from two static fields and one shared rule, with no
module aware of any other module's plan.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public instance
above; the start/target sets and the simulation itself belong to the evaluator.
