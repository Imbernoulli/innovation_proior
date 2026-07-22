# Blackstart Triage: Proactive Load Shedding under Rolling Line Trips

## Story
A transmission grid is a ring of `L` lines. Line `l` carries a real power **load**
`load[l]` and fails the instant its load exceeds its **capacity** `cap[l]`. A published
**schedule** of rolling line trips knocks lines out one per step. When a line goes dark
its load does not vanish — it is **redistributed equally** onto its still-live ring
neighbours. If that pushes a neighbour past its own cap, that line trips too, and the
failure **cascades**. A tripped line dumps its *whole* overloaded load, so an unchecked
cascade races around the ring. Power a dead line was carrying that no live neighbour can
absorb is **lost** as unmet demand.

Your only lever is **proactive shedding**: each step you may curtail load from lines.
Curtailed power is unmet demand too — but shedding early opens the **margin** a line
needs to *absorb* an incoming wave instead of tripping and dumping downstream. You cannot
shed unlimited load at once: each step has a **budget** on total curtailment, so opening a
big margin means shedding *ahead* of the wave, across several steps.

## Task
Write a **standalone program**: read ONE JSON instance from `stdin`, write ONE JSON
answer to `stdout`.

### Public instance (stdin)
```json
{ "name":"corr1", "L":16, "T":8,
  "cap":[...L ints...], "load0":[...L numbers, 0<=load0[l]<=cap[l]...],
  "nbr":[[...neighbours of line l...] ...],
  "schedule":[e_0,...,e_{T-1}],        // line tripped at step t, or -1 for none
  "alpha":0.5, "beta":1.0, "scale":40.0, "budget":18.0 }
```

### Answer (stdout)
```json
{ "shed": [ [s_{0,0},...,s_{0,L-1}], ..., [s_{T-1,0},...,s_{T-1,L-1}] ] }
```
`s_{t,l} >= 0` is the load curtailed from line `l` at the **start** of step `t` (before
that step's scheduled trip). `shed` must be exactly `T` rows of exactly `L` finite,
non-negative numbers.

### Validity
Any violation — wrong shape, a non-number / NaN / inf / negative / boolean entry, a
crash, a timeout, or non-JSON — scores **0.0** on that instance.

## Transition (per step `t`, deterministic)
1. **Curtail.** Let `raw` be the total requested shed over live lines. If `raw > budget`
   all sheds this step are scaled by `budget/raw`. Each effective shed (capped at the
   line's load) is removed from that line and added to this step's unmet demand `u`.
2. **Trip.** If `e_t` is a live line, it trips.
3. **Cascade.** While a line `x` is tripped: it goes dark; its load is split **equally**
   among its currently-live neighbours; any neighbour pushed over its cap trips next
   (carrying its *full* load). A tripped line with no live neighbour loses its load to `u`.
4. **Backlog.** `B_t = B_{t-1} + u`; add `alpha*B_t + beta*B_t^2/scale` to the episode cost.

Backlog is **monotone and carries over**: an early loss is charged on every remaining
step, and the cost is **convex** in `B_t`, so one large cascade loss hurts far more than
the same units shed a little at a time and never re-incurred. **Lower episode cost is
better.**

## Objective & scoring (deterministic)
Higher score = less backlog cost. Per instance the evaluator computes:

- `P_cand` — your plan's episode cost,
- `P_base` — the episode cost of **doing nothing** (never shedding).

```
r = clamp( 0.1 + 0.9 * (1 - P_cand / P_base), 0, 1 )
```
Doing nothing scores exactly `0.1`; a (generally unreachable) zero-backlog plan scores
`1.0`; a plan worse than doing nothing scores below `0.1`. Every scheduled trip is built
so its live neighbours' combined margin is **smaller** than its load, so some unmet
demand is **unavoidable** and even an optimal plan keeps `r < 1`. The final score is the
mean of `r` over **10** fixed seeded instances (fragile-corridor traps, greedy-friendly
diffuse grids, and twin-corridor held-out cases).

## Why it is open-ended
Relieving whichever line is most stressed **right now** is a trap: on a fragile-corridor
instance the most-utilised line is an unrelated decoy, while the wave hits a low-margin
corridor elsewhere — and because the budget is small, the margin there has to be opened
*before* the wave arrives. Many strategies trade off differently (reactive relief,
uniform pre-shedding, schedule-aware frontier shedding, full multi-step planning of *when*
and *where* to curtail against the convex carryover cost); there is no easy optimum.

## Isolation
Your program runs in a fresh sandboxed subprocess and only ever sees the public instance
above. The do-nothing reference is computed by the evaluator process.
