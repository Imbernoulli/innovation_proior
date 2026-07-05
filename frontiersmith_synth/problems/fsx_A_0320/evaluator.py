#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0320 -- "Harbor Container Port: The Secured-Zone Cordon"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A container port must fence off ONE contiguous "secured zone" inside its
rectangular storage yard.  The yard is an H x W grid of cells.  Each cell holds a
container stack with an integer VALUE v[i][j]:
    * high positive  -> a valuable export stack we WANT inside the cordon,
    * near zero / mildly negative -> low-grade cargo,
    * strongly negative -> a hazmat / reefer stack that is costly to keep secured.
Cordoning a set S of cells earns the sum of the values inside S but costs a fence:
FENCE_COST = lam per unit of PERIMETER, where the perimeter is the number of unit
grid edges separating a chosen cell from a non-chosen cell OR from the yard's
outer boundary.  The port authority wants a SINGLE connected secured zone (you
cannot fence two separate compounds) that MAXIMIZES

        profit(S) = sum_{c in S} v[c]  -  lam * perimeter(S).

This is a rectilinear-region / prize-collecting selection skinned as a port cordon
(the AtCoder-heuristic "select a polygon region under a deterministic contest
scorer" idea, made a STATIC reproducible instance).  It is NP-hard: region growing,
best-of-N seeded restarts, and add/remove local search are all viable strategies,
and no simple rule is optimal.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "H": int, "W": int, "lam": int,
             "grid": [[v_00,...,v_0{W-1}], ..., [v_{H-1}0,...]]}   # integer values
  stdout: ONE JSON object:
            {"cells": [[i,j], ...]}      # the cells inside the secured zone
          Rows i in [0,H), cols j in [0,W).  Cells must be DISTINCT, NON-EMPTY,
          and form a SINGLE 4-connected region (up/down/left/right adjacency).

  A cordon is VALID iff `cells` is a non-empty list of distinct in-bounds integer
  pairs forming one 4-connected component.  Anything else -- wrong shape, a repeat,
  an out-of-bounds pair, a disconnected set, an empty set, a crash, a timeout, or
  non-JSON -- scores 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance we compute two references,
computed by THIS parent process (the candidate never sees them):
    base = max(0, max_cell (v[c] - 4*lam))     # best SINGLE-cell cordon (weak anchor)
    ub   = sum_cell max(0, v[c] - 2*lam)        # optimistic per-cell upper bound
                                                #   (assumes an interior cell exposes
                                                #    only 2 fence edges; unreachable)
  and normalize the candidate's profit with an affine anchor
  (single-cell weak cordon -> 0.1, optimistic ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (profit_cand - base) / max(1e-9, ub - base), 0, 1 )
  A candidate that just fences the single best stack scores ~0.1; matching the
  (generally unreachable) optimistic bound scores 1.0; doing worse than the single
  best stack scores < 0.1.  Because `ub` ignores connectivity and boundary reality,
  even strong region-growing + local search stays well below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a FRESH SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
are computed HERE in the parent, so a frame-walking / filesystem-snooping candidate
learns nothing useful.

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

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_grid(seed, H, W, dist):
    """Deterministic yard of integer container-stack values."""
    ni = _rng(seed)
    v = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            r = ni(0, 99)
            if r < 12:
                v[i][j] = -ni(2, 7)          # hazmat / costly stack
            elif r < 30:
                v[i][j] = ni(4, 10)          # valuable export stack
            elif r < 55:
                v[i][j] = ni(1, 3)           # low-grade cargo
            else:
                v[i][j] = -ni(0, 2)          # near-zero / mild negative
    if dist == "moat":                        # hazmat lanes that split the yard
        for k in range(1, H, 4):
            for j in range(W):
                v[k][j] = -ni(3, 7)
    return v


def _build_instances():
    """Deterministic instance family. (seed, H, W, lam, dist)."""
    specs = [
        (101, 12, 12, 1, "moat"),
        (102, 14, 12, 1, "moat"),
        (103, 12, 14, 1, "plain"),
        (104, 14, 14, 1, "moat"),
        (105, 13, 13, 1, "plain"),
        (106, 15, 13, 1, "moat"),
        (107, 12, 16, 1, "plain"),
        (108, 16, 14, 1, "moat"),
        # harder / larger held-out instances
        (109, 15, 15, 1, "moat"),
        (110, 16, 16, 1, "plain"),
    ]
    out = []
    for seed, H, W, lam, dist in specs:
        grid = _build_grid(seed, H, W, dist)
        out.append({"name": f"yard{seed}", "H": H, "W": W, "lam": lam,
                    "grid": grid, "dist": dist})
    return out


# ----------------------------- references ----------------------------------
def _base(inst):
    v, H, W, lam = inst["grid"], inst["H"], inst["W"], inst["lam"]
    best = -10 ** 9
    for i in range(H):
        for j in range(W):
            val = v[i][j] - 4 * lam
            if val > best:
                best = val
    return max(0, best)


def _ub(inst):
    v, H, W, lam = inst["grid"], inst["H"], inst["W"], inst["lam"]
    return sum(max(0, v[i][j] - 2 * lam) for i in range(H) for j in range(W))


# ----------------------------- validation + objective ----------------------
def _profit(inst, answer):
    """Validate the cordon. Return profit (int) if valid, else None."""
    if not isinstance(answer, dict):
        return None
    cells = answer.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    H, W = inst["H"], inst["W"]
    v, lam = inst["grid"], inst["lam"]
    seen = set()
    pts = []
    for c in cells:
        if not (isinstance(c, list) and len(c) == 2):
            return None
        i, j = c
        if isinstance(i, bool) or isinstance(j, bool):
            return None
        if not (isinstance(i, int) and isinstance(j, int)):
            return None
        if not (0 <= i < H and 0 <= j < W):
            return None
        if (i, j) in seen:                    # duplicate cell -> invalid
            return None
        seen.add((i, j))
        pts.append((i, j))
    # 4-connectivity: single component
    start = pts[0]
    comp = {start}
    stack = [start]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if n in seen and n not in comp:
                comp.add(n)
                stack.append(n)
    if len(comp) != len(seen):
        return None
    # perimeter = exposed unit edges
    per = 0
    for (i, j) in pts:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (i + di, j + dj) not in seen:
                per += 1
    total = sum(v[i][j] for (i, j) in pts)
    return total - lam * per


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        base = _base(inst)
        ub = _ub(inst)
        denom = ub - base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "H": inst["H"], "W": inst["W"],
                  "lam": inst["lam"], "grid": [row[:] for row in inst["grid"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            profit = _profit(inst, ans)
        except Exception:
            profit = None
        if profit is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (profit - base) / denom
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
