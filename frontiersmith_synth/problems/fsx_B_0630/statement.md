# Morning Rush: Twin-Car Shaft Dispatch

A tower's lobby has `S` elevator **shafts**. Each shaft holds **two cars
sharing one hoistway**: a lower-role car and an upper-role car. They can
never pass each other; at every tick the upper car must stay at least `G`
floors above the lower (`floor(upper) - floor(lower) >= G`). Cars move one
floor per tick; when idle (nobody waiting for it, nobody aboard) a car
drifts toward its assigned **park floor** — also its floor at tick 0. A car
sweeps in its current direction, boarding any arrived passenger it passes
and dropping off anyone aboard whose floor it reaches, before deciding to
continue, reverse, or (if idle) head to its park floor.

You write a **dispatch plan** for one fixed morning-rush trace: which
(shaft, role) car serves each hall call, and where every car parks. The
evaluator then replays the whole shift deterministically and scores the sum
of squared passenger waiting times.

## Candidate program contract

Standalone program: read ONE JSON public instance from **stdin**, write ONE
JSON answer to **stdout**. Runs isolated; sees only the public instance.

```python
import sys, json
inst = json.load(sys.stdin)
# ... compute ...
print(json.dumps({"assign": assign, "park": park}))
```

### Public instance (stdin)

```json
{
  "F": 16,          // floors, 0..F-1
  "S": 2,           // number of shafts (2 cars each -> 2*S cars total)
  "G": 2,           // minimum floor gap between a shaft's two cars
  "T": 300,         // simulation horizon (ticks)
  "W_cap": 300,     // wait charged to a call never picked up by T
  "calls": [ {"id": 0, "t": 3, "o": 5, "d": 12}, ... ]  // arrival tick, origin, dest
}
```

All calls for the whole morning are given up front (you know the full
demand mix); the physics — sweep direction, boarding, the non-passing gap —
is simulated by the evaluator, not by you.

### Answer (stdout)

```json
{
  "assign": [[0, 0], [1, 1], ...],   // one [shaft, role] per call, role 0=lower 1=upper
  "park": [0, 15, 0, 15]             // length 2*S: park[2s]=lower car of shaft s, park[2s+1]=upper
}
```

`assign` must have exactly one `[shaft, role]` entry per call (`0<=shaft<S`,
`role in {0,1}`); `park` must have exactly `2*S` integer floors in
`[0, F-1]`. The two park floors of every shaft must already satisfy the
gap (`park[2s+1] - park[2s] >= G`) — this is also enforced continuously
during the simulated shift. Any violation, wrong shape, non-integer, a
crash, timeout, or non-JSON output scores that instance `0.0`.

## Objective

**Minimize**, over a fixed seeded family of 10 morning-rush traces, the sum
of squared waiting times (arrival tick to pickup tick; an unpicked call
costs `W_cap^2`). Several traces cluster demand away from the building's
extremes or near the shared safety gap — a car that always reacts call by
call and idles at a fixed default floor gets dragged toward the same busy
zone as its shaft partner, where the non-passing gap serializes them.

## Scoring (deterministic)

For each instance the evaluator computes, itself, the objective `obj` of an
internal weak reference construction `b` (pure round-robin across all
`2*S` cars by arrival index — not even a fixed floor split, fully blind to
where any call actually is — park at the two extremes), then scores your
plan's objective `q`:

```
r = min(1.0, 0.1 * b / max(q, 1e-12))
```

Matching the weak reference scores `~0.1`; a plan `10x` better in summed
squared wait scores `~1.0`. **Ratio** is the mean of `r` over 10 instances;
**Vector** lists the per-instance scores.

## Suggested strategies

1. **Blind round robin**: cycle every call across all `2*S` cars by arrival
   index, ignoring floors; park at the extremes.
2. **Nearest-car greedy**: assign each call, in arrival order, to whichever
   car predicts the shortest distance to the origin — no pre-planning, no
   re-parking, no check that the target floor is even reachable given the
   safety gap to its shaft partner.
3. **Demand-tuned floor bands**: look at the whole demand mix once, slice it
   into per-shaft, per-role bands separated by at least the gap, and re-park
   each car at its band's center — sized off the *partner's* required reach
   so neither car can ever rest where the other must go.
4. **Local refinement**: perturb a band-partition's assignments or park
   floors to squeeze out remaining serialization stalls.
