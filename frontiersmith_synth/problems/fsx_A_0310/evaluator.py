#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0310 -- "Relay Run: Fuel-Budgeted Interstellar Survey"
(family: heuristic-contest-offline; format B, quality-metric; theme: interstellar relay).

THEME.  A survey ship carrying a deployable relay beacon launches from a fixed
STARBASE.  Scattered across a sector are N candidate star systems, each at a 2-D
position and each holding a fixed SCIENCE VALUE (a rich, far anomaly is worth far
more than a nearby quiet dwarf).  The ship flies an open route -- starbase, then an
ordered list of distinct systems -- dropping a relay beacon at each system it visits.
Its jump drive carries a fixed FUEL budget L: the total Euclidean length of the flown
path (starbase -> first system -> ... -> last system) must not exceed L.  The ship
wants to MAXIMIZE the total science value of the systems it actually reaches within
that fuel budget.

This is the classic ORIENTEERING / prize-collecting path problem skinned as an
interstellar relay survey: systems = prized nodes, the starbase = the depot, fuel =
the path-length budget, and we maximize collected prize on a single open tour.  It is
NP-hard.  A fuel-blind "nearest hop" rule that ignores prize is easily beaten; a
value-density insertion greedy does better but never revisits its choices, so it
wastes fuel on detours it can no longer undo; a warm-started local search that runs
2-opt to shorten the route and swaps cheap systems for richer reachable ones does
better still -- yet the total-prize ceiling (visit every system) is unreachable under
any real fuel budget, so scores keep headroom below 1.0.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "L": float,
             "bx": float, "by": float,            # starbase coordinates
             "x": [x_0, ...], "y": [y_0, ...],    # system coordinates, length N
             "p": [p_0, ...]}                      # system science value, length N, p >= 0
  stdout: ONE JSON object:
            {"route": [i0, i1, ...]}
          the ORDERED list of DISTINCT system indices to visit.  The ship flies
          starbase -> systems[i0] -> systems[i1] -> ...  An empty route is legal.

  A route is VALID iff `route` is a list of distinct integers in [0, N) whose flown
  path length (starbase to the first system, then consecutively between systems)
  is <= L + 1e-6.  A repeated index, an out-of-range index, over budget, a crash,
  a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  For each instance we compute two references
in THIS parent process:
    q_base  = collected value of the internal FUEL-BLIND NEAREST-HOP operator
              (from the starbase, always jump to the nearest unvisited system that
              still fits the fuel budget; ignores science value entirely)
    q_full  = total value of ALL systems (the visit-everything ceiling, ignoring
              fuel; generally infeasible, hence unreachable)
  Let q_cand be the collected value of the candidate's valid route.  We normalize
  with an affine anchor (fuel-blind hop -> 0.1, everything -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_full - q_base), 0, 1 )
  Matching the nearest-hop reference scores ~0.1; collecting every system's value
  scores 1.0; doing worse than the nearest hop scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`, seeing ONLY the public instance.  The references (q_base,
q_full) are computed by this parent, so a frame-walking / source-reading candidate
learns nothing that helps it inflate its score.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_instance(seed, N, n_rich, fuel):
    """N systems in a 1000x1000 sector; most are quiet (low value), a few are rich
    high-value anomalies placed far from the crowd, so a value-blind route wastes
    fuel on cheap near systems and misses the anomalies."""
    nxt = _rng(seed)
    bx = 500.0; by = 500.0
    x = []; y = []; p = []
    for _ in range(N):
        x.append(float(nxt(0, 1000)))
        y.append(float(nxt(0, 1000)))
        p.append(float(nxt(1, 6)))            # quiet ambient science value
    # promote a few systems to rich anomalies (much larger value)
    for _ in range(n_rich):
        j = nxt(0, N - 1)
        p[j] = float(nxt(30, 90))
    return {"name": f"sector{seed}", "N": N, "L": float(fuel),
            "bx": bx, "by": by, "x": x, "y": y, "p": p}


def _build_instances():
    """Deterministic instance family: (seed, N, n_rich, fuel)."""
    specs = [
        (3101, 34, 5, 2400),
        (3102, 38, 6, 2500),
        (3103, 40, 6, 2600),
        (3104, 34, 4, 2200),
        (3105, 44, 7, 2700),
        (3106, 38, 5, 2500),
        (3107, 40, 6, 2400),
        (3108, 44, 8, 2800),
        # harder / larger held-out sectors (more systems, tighter fuel)
        (3151, 52, 9, 2600),
        (3152, 48, 8, 2500),
        (3153, 56, 10, 2700),
        (3154, 50, 9, 2400),
    ]
    return [_build_instance(*s) for s in specs]


# ----------------------------- geometry helpers ----------------------------
def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _route_length(inst, route):
    """Flown path length: starbase -> first system -> ... -> last system."""
    if not route:
        return 0.0
    x = inst["x"]; y = inst["y"]
    total = _dist(inst["bx"], inst["by"], x[route[0]], y[route[0]])
    for k in range(len(route) - 1):
        a = route[k]; b = route[k + 1]
        total += _dist(x[a], y[a], x[b], y[b])
    return total


def _route_value(inst, route):
    p = inst["p"]
    return sum(p[j] for j in route)


# ----------------------------- references ----------------------------------
def _nearest_hop(inst):
    """Weak fuel-blind operator: from the starbase, always jump to the nearest
    unvisited system that still fits the remaining fuel; ignores science value."""
    N = inst["N"]; L = inst["L"]; x = inst["x"]; y = inst["y"]
    cx = inst["bx"]; cy = inst["by"]
    used = 0.0
    visited = [False] * N
    route = []
    while True:
        best_j = -1; best_d = None
        for j in range(N):
            if visited[j]:
                continue
            d = _dist(cx, cy, x[j], y[j])
            if used + d <= L + 1e-9 and (best_d is None or d < best_d):
                best_d = d; best_j = j
        if best_j < 0:
            break
        used += best_d
        cx = x[best_j]; cy = y[best_j]
        visited[best_j] = True
        route.append(best_j)
    return _route_value(inst, route)


def _full_ceiling(inst):
    return sum(inst["p"])


# ----------------------------- validation ----------------------------------
def _valid_route_value(inst, answer):
    """Validate the candidate answer; return collected value or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    route = answer.get("route")
    if not isinstance(route, list):
        return None
    N = inst["N"]
    seen = set()
    for j in route:
        if isinstance(j, bool) or not isinstance(j, int):
            return None
        if j < 0 or j >= N:
            return None
        if j in seen:
            return None
        seen.add(j)
    length = _route_length(inst, route)
    if not (length == length) or length in (float("inf"), float("-inf")):
        return None
    if length > inst["L"] + 1e-6:
        return None
    return _route_value(inst, route)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _nearest_hop(inst)
        q_full = _full_ceiling(inst)
        denom = q_full - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "N": inst["N"], "L": inst["L"],
                  "bx": inst["bx"], "by": inst["by"],
                  "x": list(inst["x"]), "y": list(inst["y"]), "p": list(inst["p"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _valid_route_value(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
