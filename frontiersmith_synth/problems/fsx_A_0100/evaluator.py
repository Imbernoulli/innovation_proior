#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0100 -- "Tell Qadesh: The Connected Dig Plan"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  An archaeological survey has mapped a square dig site as an N x N grid of
cells.  Ground-penetrating radar assigns every cell an integer "yield" v[r][c]:
positive where buried artifacts are expected, negative where the cell is barren
rubble that costs effort to clear.  The excavation team will dig out ONE connected
trench -- a set of cells that is 4-connected (orthogonal neighbours) so the pit can
be walked and shored as a single site.  Digging cell (r,c) collects v[r][c].  Every
exposed face of the trench must be shored: the SHORING COST is lambda per unit of
perimeter, where the perimeter is the number of grid edges between a dug cell and a
non-dug cell OR the outside of the grid.

    plan value  =  sum of v over dug cells  -  lambda * (perimeter of the dug region)

The team wants to MAXIMISE the plan value by choosing which connected trench to dig
(the empty plan is allowed and scores 0).  This is an AtCoder-heuristic-contest-style
offline optimisation: a rich landscape (a rectilinear region-selection problem, akin
to a polygon selection over the grid) with a deterministic contest scorer.  There is
no easy optimum -- unconstrained the objective is a graph-cut, but the CONNECTIVITY
requirement makes it NP-hard, so region-growing, multi-start hill-climbing, boundary
smoothing and seeded local search all give genuinely different results.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": N (int), "lam": lambda (int),
             "grid": [[v[0][0], ...], ...]}   # N rows, N ints each
  stdout: ONE JSON object:
            {"cells": [[r0,c0], [r1,c1], ...]}   # the dug cells (0-indexed)

  A plan is VALID iff "cells" is a list of [r,c] integer pairs, each in range
  0<=r,c<N, with NO duplicates, and (if non-empty) the cells form a SINGLE
  4-connected region.  A disconnected set, an out-of-range or duplicate cell, a
  non-integer coordinate, a crash, a timeout, or non-JSON -> that instance scores 0.0.
  The empty plan [] is valid (value 0).

SCORING (deterministic; no wall-time).  Per instance the evaluator computes:
    base = value of the single best cell as a size-1 trench
           = max_c ( v[c] - 4*lambda )                        # weak reference (>0 by design)
    ub   = sum of all strictly-positive yields, with NO shoring cost
           = sum_{v[c]>0} v[c]                                 # optimistic, unreachable bound
    cand = plan value achieved by the candidate's trench
  and normalises with an affine anchor (weak reference -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - base) / (ub - base), 0, 1 )
  A candidate that only digs the best single cell scores ~0.1; reaching the (generally
  unreachable) all-artifacts bound scores 1.0; doing worse than the best single cell
  scores < 0.1.  Because `ub` ignores connectivity AND shoring, even excellent trenches
  stay well below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(base, ub) and the connectivity/validity check are computed by THIS parent process,
so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_grid(seed, N, n_hot, gap):
    """Deterministic N x N integer yield grid: a barren negative background with
    `n_hot` positive artifact hotspots (peaked bumps).  `gap` controls how far
    hotspot centres are pushed apart (larger gap -> harder to bridge)."""
    ni = _rng(seed)
    # barren background: small clearing cost
    grid = [[-ni(1, 3) for _ in range(N)] for _ in range(N)]
    centers = []
    tries = 0
    while len(centers) < n_hot and tries < 400:
        tries += 1
        cr = ni(2, N - 3)
        cc = ni(2, N - 3)
        ok = True
        for (pr, pc) in centers:
            if abs(pr - cr) + abs(pc - cc) < gap:
                ok = False
                break
        if ok:
            centers.append((cr, cc))
    for (cr, cc) in centers:
        peak = ni(35, 85)
        rad = ni(2, 4)
        for r in range(max(0, cr - rad), min(N, cr + rad + 1)):
            for c in range(max(0, cc - rad), min(N, cc + rad + 1)):
                d = abs(r - cr) + abs(c - cc)
                if d <= rad:
                    add = int(round(peak * (1.0 - d / (rad + 1.0))))
                    if add > 0:
                        grid[r][c] += add
    return grid


def _build_instances():
    """Deterministic instance family. (seed, N, n_hot, gap, lam)."""
    specs = [
        (101, 16, 3, 5, 2),
        (102, 18, 3, 6, 2),
        (103, 18, 4, 5, 3),
        (104, 20, 3, 8, 2),   # far hotspots: bridging rarely pays
        (105, 20, 4, 5, 2),   # near hotspots: bridging can pay
        (106, 22, 4, 6, 3),
        (107, 22, 5, 5, 2),
        (108, 24, 4, 7, 3),
        # harder / larger held-out instances
        (211, 26, 5, 6, 3),
        (212, 26, 6, 5, 2),
        (213, 28, 6, 6, 3),
        (214, 28, 5, 8, 2),
    ]
    out = []
    for seed, N, n_hot, gap, lam in specs:
        grid = _build_grid(seed, N, n_hot, gap)
        out.append({"name": f"tell{seed}", "N": N, "lam": lam, "grid": grid})
    return out


# ----------------------------- references ----------------------------------
def _base(grid, lam):
    """Value of the best single-cell trench (perimeter of a lone cell = 4)."""
    best = None
    for row in grid:
        for v in row:
            s = v - 4 * lam
            if best is None or s > best:
                best = s
    return best


def _ub(grid):
    """Optimistic upper bound: collect every positive yield, pay no shoring."""
    return sum(v for row in grid for v in row if v > 0)


# ----------------------------- validation / scoring ------------------------
def _plan_value(inst, answer):
    """Validate the answer against the instance. Return the plan value (float),
    or None if the plan is infeasible / malformed."""
    if not isinstance(answer, dict):
        return None
    cells = answer.get("cells")
    if not isinstance(cells, list):
        return None
    N = inst["N"]
    grid = inst["grid"]
    lam = inst["lam"]
    dug = set()
    for item in cells:
        if not isinstance(item, list) or len(item) != 2:
            return None
        r, c = item
        if isinstance(r, bool) or isinstance(c, bool):
            return None
        if not isinstance(r, int) or not isinstance(c, int):
            return None
        if r < 0 or r >= N or c < 0 or c >= N:
            return None
        if (r, c) in dug:
            return None                       # duplicate cell
        dug.add((r, c))
    if not dug:
        return 0.0                            # empty plan is valid, value 0
    # connectivity: exactly one 4-connected component
    start = next(iter(dug))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in dug and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(dug):
        return None                           # disconnected -> infeasible
    # value = sum(v) - lam * perimeter
    total = 0
    perim = 0
    for (r, c) in dug:
        total += grid[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb not in dug:
                perim += 1                     # face touching non-dug cell or outside
    return float(total - lam * perim)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        grid = inst["grid"]
        lam = inst["lam"]
        base = _base(grid, lam)
        ub = _ub(grid)
        denom = ub - base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "N": inst["N"], "lam": lam,
                  "grid": [list(row) for row in grid]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            val = _plan_value(inst, ans)
        except Exception:
            val = None
        if val is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (val - base) / denom
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
