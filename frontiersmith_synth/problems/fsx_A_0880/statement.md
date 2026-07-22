# Container Yard: Corridor Reservation Dispatch

A container terminal's yard is a **ring** of `n_nodes` junctions connecting the
stack blocks — a wide perimeter lane where any number of AGVs (automated guided
vehicles) can pass each other freely. A few pairs of junctions are additionally
linked by a **shortcut aisle**: a straight cut between stack blocks that is much
shorter than walking around the ring, but only **one AGV at a time** fits in it
(it is a single-lane gap). You must dispatch a batch of move requests — each an
AGV that must travel from a source junction to a destination junction, no earlier
than a given release time — maximizing completed throughput.

## Instance (stdin, one JSON object)

```
{"name": str, "n_nodes": int,
 "edges": [{"id": int, "u": int, "v": int, "length": int, "shared": bool}, ...],
 "moves": [{"id": int, "src": int, "dst": int, "release": number}, ...],
 "horizon": number}
```
Every edge is traversable in either direction and takes `length` time units to
cross. `shared: true` marks a single-lane shortcut aisle; all other edges (the
ring) have unlimited capacity.

## Answer (stdout, one JSON object)

```
{"moves": [{"id": int, "path": [n0, n1, ..., nk], "times": [t0, t1, ..., tk]}, ...]}
```
For each entry, `path` must start at that move's `src` and end at its `dst`,
and every consecutive pair `(path[i], path[i+1])` must be a real edge. `times`
gives the instant the AGV is present at each node of `path`: `t0 >= release`,
and for every edge, `times[i+1] - times[i] >= length` of that edge (an AGV may
wait at a node before continuing, but never crosses an edge faster than its
length). A move whose `id` is **omitted** from the list is simply not attempted
(0 credit for it, no penalty elsewhere). A move whose entry is malformed,
disconnected, out of range, or non-finite is invalid and scores 0 **for that
move only** — it does not disqualify the rest of your answer. If a move's
finish time exceeds `horizon`, it also scores 0.

## The single-lane rule (why coordination matters)

Record every AGV's occupancy window `[times[i], times[i+1])` on every shared
edge it crosses, together with its direction of travel. On a given shared edge,
if two AGVs' windows **overlap** in time:
* both of those two moves fail (score 0), and
* if they were traveling in **opposite directions**, that is a head-on
  **deadlock**: from the *later* of the two windows' start times onward, the
  aisle is jammed — **every** occupancy window on that edge (from any move,
  including ones you schedule to depart afterward) that starts at or after
  that instant also fails. One head-on collision can silently strand a whole
  chain of later moves that individually looked fine.

Racing every AGV down its shortest path the instant it is released is the
obvious recipe — and it is exactly what triggers these head-ons when two
requests naturally cross the same shortcut from opposite ends. Reserving each
shortcut's time-windows in advance — delaying an AGV's entry until the aisle is
clear, or occasionally sending it the long way around the ring instead — costs
a little time but guarantees no head-on ever happens.

## Scoring

A surviving move contributes
`1 - 0.35 * max(0, finish - ideal_finish) / ideal_finish` (clipped to `[0,1]`),
where `ideal_finish = release + shortest-path time ignoring all traffic`; a
failed or unattempted move contributes `0`. Summing over all moves in the
instance gives your raw score. The same formula applied to the *naive* plan
(every AGV takes its graph-shortest path and departs immediately at release)
gives a reference `raw_base`. Your per-instance ratio is

```
r = clamp(0.1 + 0.9 * (raw_you - raw_base) / (1.15*M - raw_base), 0, 1)
```

where `M` is the number of moves in the instance. Matching the naive plan
scores about `0.1`; doing worse scores lower (down to `0`); resolving
conflicts the naive plan deadlocked on scores substantially higher. Your
overall score is the mean `r` over 12 seeded instances. Deterministic,
no wall-clock dependence: identical output on identical input always
produces identical score.
