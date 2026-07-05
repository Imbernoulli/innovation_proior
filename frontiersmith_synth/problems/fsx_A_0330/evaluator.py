#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0330 -- "Alpine Lift Circuit: Overnight Inspection Route"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A large ski resort must inspect every lift station on the mountain overnight.
A single snowcat departs the base depot, drives to each of the N lift stations exactly
once (in some order), and returns to the depot.  Each station has a 3-D map position
(x, y, elevation).  The snowcat's travel cost between two stations is the rounded
Euclidean distance in this 3-D map (horizontal distance plus vertical climb), so
elevation swings are expensive.

The resort MINIMIZES the total length of the closed inspection circuit
        cost(order) = d(depot, s_1) + sum_i d(s_i, s_{i+1}) + d(s_last, depot)
where d(a,b) = round( sqrt((xa-xb)^2 + (ya-yb)^2 + (za-zb)^2) ).

This is an offline AtCoder-heuristic-style routing contest (a metric TSP over lift
stations) scored by a fixed deterministic formula under a fixed local-search
op-budget -- NOT wall-time.  The interesting tension: the "do-nothing" order (visit
stations in the order they appear on the maintenance roster) is a terrible circuit;
a greedy nearest-station chain is far better; and greedy + 2-opt edge-uncrossing does
better still, but the loose lower bound keeps even excellent routes below 1.0.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": N (int),                 # number of lift stations (excludes depot)
             "coords": [[x,y,z], ...]}      # length N+1; index 0 = base depot,
                                            # indices 1..N = the lift stations
  stdout: ONE JSON object:
            {"order": [i_1, i_2, ..., i_N]}   # a PERMUTATION of the integers 1..N
          giving the sequence in which the snowcat visits the stations.  The circuit
          is depot -> i_1 -> ... -> i_N -> depot.

  An answer is VALID iff `order` is a list of exactly N integers that is a permutation
  of {1, 2, ..., N} (each station visited exactly once, no depot, no repeats, nothing
  out of range).  Wrong length, a repeat, a missing/extra index, a non-integer, a
  crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    b   = cost of the ROSTER order [1, 2, ..., N] (the do-nothing baseline circuit).
    obj = cost of the candidate's circuit.
  normalized (minimization; roster baseline -> 0.1, and a 10x-shorter circuit -> 1.0):
    r = clamp( 0.1 * b / max(obj, 1e-9), 0, 1 )
  Reproducing the roster order scores exactly 0.1; a longer circuit scores < 0.1;
  shorter circuits score higher.  Because no metric TSP circuit can beat its own tight
  lower bound, and the roster/optimal ratio is bounded on these instances, strong
  routes stay comfortably below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The baseline cost and
all validation happen in THIS parent process, so a frame-walking / introspecting
candidate learns nothing that helps it shorten its route.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- instance family -----------------------------
def _build_instance(seed, n):
    """Deterministic instance: base depot + N lift stations on a mountain map.

    Stations are drawn in ROSTER order (the order they are listed), which is NOT
    spatially sorted, so the roster circuit [1..N] is a poor baseline.  Elevation
    z is correlated with distance from the summit ridge to look mountain-like, but
    the roster order ignores geometry entirely.
    """
    rng = random.Random(seed)
    coords = []
    # base depot sits low, near a corner of the map
    coords.append([rng.randint(0, 120), rng.randint(0, 120), rng.randint(0, 60)])
    # a couple of "summit" attractors to give the elevation field structure
    summits = [(rng.randint(0, 1000), rng.randint(0, 1000)) for _ in range(3)]
    for _ in range(n):
        x = rng.randint(0, 1000)
        y = rng.randint(0, 1000)
        # elevation: higher when close to a summit attractor, plus noise
        best = min((sx - x) ** 2 + (sy - y) ** 2 for (sx, sy) in summits)
        z = int(max(0, 500 - math.isqrt(best) // 2)) + rng.randint(0, 80)
        coords.append([x, y, z])
    return {"name": f"resort{seed}", "n": n, "coords": coords}


def _build_instances():
    """Deterministic instance family (large-scale metric TSP over lift stations).

    A spread of station counts; the last few are larger held-out instances that
    stress generalization of the search heuristic.
    """
    specs = [
        (3101, 60), (3102, 70), (3103, 75), (3104, 80),
        (3105, 85), (3106, 90), (3107, 95), (3108, 100),
        # larger / harder held-out
        (3211, 110), (3212, 120), (3213, 130), (3214, 140),
    ]
    return [_build_instance(seed, n) for (seed, n) in specs]


# ----------------------------- references / scoring ------------------------
def _dist(a, b):
    dx = a[0] - b[0]; dy = a[1] - b[1]; dz = a[2] - b[2]
    return int(round(math.sqrt(dx * dx + dy * dy + dz * dz)))


def _circuit_cost(coords, order):
    """Closed circuit cost depot(0) -> order -> depot(0)."""
    total = _dist(coords[0], coords[order[0]])
    for i in range(len(order) - 1):
        total += _dist(coords[order[i]], coords[order[i + 1]])
    total += _dist(coords[order[-1]], coords[0])
    return total


def _validate(inst, answer):
    """Validate the route. Return the order list (ints) or None."""
    if not isinstance(answer, dict):
        return None
    order = answer.get("order")
    if not isinstance(order, list):
        return None
    n = inst["n"]
    if len(order) != n:
        return None
    seen = [False] * (n + 1)
    for v in order:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 1 or v > n:
            return None
        if seen[v]:
            return None
        seen[v] = True
    return order


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        coords = inst["coords"]
        n = inst["n"]
        b = _circuit_cost(coords, list(range(1, n + 1)))   # roster baseline
        public = {"name": inst["name"], "n": n,
                  "coords": [list(p) for p in coords]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            order = _validate(inst, ans)
        except Exception:
            order = None
        if order is None:
            vec.append(0.0)
            continue
        obj = _circuit_cost(coords, order)
        r = 0.1 * b / max(obj, 1e-9)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
