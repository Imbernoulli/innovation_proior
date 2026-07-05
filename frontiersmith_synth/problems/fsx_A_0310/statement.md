# Relay Run: Fuel-Budgeted Interstellar Survey

## Story

A survey ship carrying a deployable relay beacon launches from a fixed **starbase**.
Scattered across a sector are `N` candidate **star systems**, each at a 2-D position
and each holding a fixed **science value** (a rich, far anomaly is worth far more than
a nearby quiet dwarf). The ship flies an **open route** — starbase, then an ordered
list of distinct systems — dropping a relay beacon at each system it visits. Its jump
drive carries a fixed **fuel budget** `L`: the total Euclidean length of the flown
path (starbase → first system → … → last system) must not exceed `L`. Maximize the
total science value of the systems you actually reach within that fuel budget.

This is the classic **orienteering / prize-collecting path** problem, skinned as an
interstellar relay survey. It is NP-hard. A fuel-blind "nearest hop" rule that ignores
prize is easily beaten; a value-density insertion greedy does better but never revisits
its choices; a warm-started local search (2-opt to shorten the route, plus insertion
and drop-and-swap moves) does better still — yet the visit-everything ceiling is
unaffordable under any real fuel budget, so scores keep headroom below 1.0.

## Candidate program contract

Your program is a **standalone stdin → stdout process**. It reads ONE JSON object (the
public instance) and writes ONE JSON object (the route).

### Input (stdin) — the PUBLIC instance

```json
{
  "name": "sector3101",
  "N": 34,
  "L": 2400.0,
  "bx": 500.0, "by": 500.0,
  "x": [x_0, x_1, ...],
  "y": [y_0, y_1, ...],
  "p": [p_0, p_1, ...]
}
```

- `N` — number of candidate star systems.
- `L` — fuel budget (maximum flown path length).
- `bx, by` — starbase coordinates (the route starts here; no beacon dropped, no value).
- `x[i], y[i]` — coordinates of system `i` (all in `[0, 1000]`).
- `p[i]` — science value of system `i` (`p[i] >= 0`).

### Output (stdout) — the route

```json
{"route": [i0, i1, ...]}
```

The ordered list of **distinct** system indices to visit. The ship flies
`starbase → systems[i0] → systems[i1] → …`. An **empty route** (`{"route": []}`) is
legal and scores the fuel-blind baseline anchor (~0 collected value).

## Feasibility (a route scores 0.0 otherwise)

A route is **valid** iff:

- `route` is a list of **distinct** integers, each in `[0, N)`, and
- the flown path length — `dist(starbase, systems[route[0]])` plus the consecutive
  distances between visited systems — is `<= L + 1e-6`.

A repeated index, an out-of-range index, an over-budget route, a crash, a timeout, or
non-JSON output makes that instance score **0.0**.

## Objective and scoring (deterministic)

Let `q_cand` be the total science value collected by your valid route
(`sum of p[i] for i in route`). For each instance the evaluator computes two
references **in the judge process** (never sent to you):

- `q_base` — value collected by the internal **fuel-blind nearest-hop** operator
  (from the starbase, always jump to the nearest unvisited system that still fits the
  remaining fuel; ignores science value).
- `q_full` — total value of **all** systems (the visit-everything ceiling, ignoring
  fuel; generally infeasible).

Your per-instance score is an affine anchor (weak nearest-hop → 0.1, everything → 1.0):

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_full - q_base), 0, 1 )
```

The reported **Ratio** is the mean of `r` over all instances (a mix of base and
harder, larger held-out sectors). Matching the nearest-hop reference scores ~0.1;
collecting every system's value scores 1.0; doing worse than the nearest hop scores
below 0.1.

## Isolation

Your program runs in a fresh **sandboxed subprocess** and sees only the public instance
above. The references and the scoring live in the judge process, which the sandbox
cannot reach, so introspection or source-reading gains you nothing — only a better
route raises your score.
