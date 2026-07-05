#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0110 -- "Polar Research Base: Footprint Selection"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A survey drone has mapped a rectangular sector of an antarctic ice sheet
into an H x W grid of cells.  Each cell carries an INTEGER "net terrain value":
  * high positive cells are science hotspots (buried meteorites, subglacial lakes,
    ice cores worth drilling);
  * mildly negative cells are ordinary ice (a small build/levelling cost);
  * strongly negative cells are crevasses / pressure ridges you would rather avoid.
You must stake out the base FOOTPRINT: a non-empty, 4-connected set of cells the
station will occupy.  Because the polar plateau is battered by wind, every unit of
the footprint's exposed OUTER BOUNDARY (an edge separating a chosen cell from an
unchosen cell or from the sector border) must be insulated/wind-shielded at a fixed
cost `perim_cost` per unit edge.

This reframes an AtCoder-heuristic-style "select a scored polygon on a fixed
instance" contest as a STATIC, deterministically-scored offline problem: pick a
connected region of grid cells to MAXIMIZE
        net = (sum of chosen cell values) - perim_cost * (boundary length).
Maximum-weight *connected* subgraph selection with a boundary penalty is NP-hard,
so there is no easy optimum -- greedy region-growing, multi-seed growth, and
seeded add/anneal local search all trade off differently (open-ended).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "H": int, "W": int, "perim_cost": int,
             "grid": [[v_00, ...], ...]}   # H rows x W cols of integer values
  stdout: ONE JSON object:
            {"cells": [[r0,c0], [r1,c1], ...]}   # the chosen footprint
          Each [r,c] is a grid coordinate (0<=r<H, 0<=c<W).  The list must be
          NON-EMPTY, contain no duplicates, and form a single 4-connected region.

  A footprint is VALID iff `cells` is a non-empty list of in-range, distinct
  integer [r,c] pairs that are 4-connected.  Invalid output, a disconnected set,
  an out-of-range/duplicate cell, a crash, a timeout, or non-JSON -> instance
  scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute:
    b  = weak baseline net = best SINGLE-cell footprint
         = max_over_cells(v) - 4*perim_cost   (a lone cell has 4 boundary edges)
    U  = loose upper bound = sum of all STRICTLY-POSITIVE cell values
         (ignores connectivity AND the perimeter penalty -> generally unreachable)
    c  = candidate footprint net value
  and normalize with an affine anchor (weak baseline -> 0.1, ideal U -> 1.0):
    r = clamp( 0.1 + 0.9 * (c - b) / max(1e-9, U - b), 0, 1 )
  Reproducing the single-cell baseline scores ~0.1; approaching the (unreachable)
  positive-mass bound approaches 1.0; a worse-than-baseline footprint scores < 0.1.
  Because U ignores both connectivity and the perimeter cost, even strong local
  search stays strictly below 1.0 -> headroom preserved.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(baseline b, bound U) and the connectivity/validity check are computed by THIS
parent process, so a frame-walking / introspecting candidate learns nothing.

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
def _build_grid(seed, H, W, n_bump, n_crev):
    """Deterministic integer terrain grid.  A mild-negative ice base, plus several
    smooth POSITIVE science "bumps" (linear-falloff plateaus that can overlap into
    ridges), minus a few sharp NEGATIVE crevasse bumps.  This yields smooth value
    fields, so a footprint's best shape is a non-obvious compact super-level set
    rather than a scatter of isolated spikes -> region growth has real traction and
    connecting neighbouring plateaus genuinely pays off."""
    ni = _rng(seed)
    grid = [[-1 for _ in range(W)] for _ in range(H)]   # mild ice cost base

    def _stamp(count, amp_lo, amp_hi, rad_lo, rad_hi, sign):
        for _ in range(count):
            cr = ni(0, H - 1)
            cc = ni(0, W - 1)
            amp = ni(amp_lo, amp_hi)
            rad = ni(rad_lo, rad_hi)
            for r in range(max(0, cr - rad), min(H, cr + rad + 1)):
                for c in range(max(0, cc - rad), min(W, cc + rad + 1)):
                    d = abs(r - cr) + abs(c - cc)       # manhattan falloff
                    if d <= rad:
                        contrib = (amp * (rad + 1 - d)) // (rad + 1)
                        grid[r][c] += sign * contrib

    _stamp(n_bump, 9, 17, 2, 5, +1)    # science plateaus (positive)
    _stamp(n_crev, 8, 20, 1, 3, -1)    # crevasses (negative)
    return grid


def _build_instances():
    """Deterministic instance family. (seed, H, W, n_bump, n_crev, perim_cost)."""
    specs = [
        (101, 20, 20, 6, 6, 2),
        (102, 22, 22, 7, 7, 2),
        (103, 24, 20, 7, 8, 1),
        (104, 20, 24, 6, 6, 3),
        (205, 24, 24, 8, 9, 2),
        (206, 22, 26, 8, 8, 2),
        (107, 26, 22, 9, 10, 3),
        (108, 24, 24, 8, 8, 1),
        # harder / larger held-out sectors (more, more-spread bumps)
        (311, 30, 28, 11, 14, 2),
        (312, 28, 32, 12, 16, 3),
        (313, 32, 30, 13, 16, 2),
        (314, 34, 30, 14, 18, 3),
    ]
    out = []
    for seed, H, W, n_bump, n_crev, pc in specs:
        grid = _build_grid(seed, H, W, n_bump, n_crev)
        out.append({"name": f"sector{seed}", "H": H, "W": W,
                    "perim_cost": pc, "grid": grid})
    return out


# ----------------------------- references ----------------------------------
def _baseline(inst):
    """Weak baseline net value = best single-cell footprint."""
    pc = inst["perim_cost"]
    best = None
    for row in inst["grid"]:
        for v in row:
            net = v - 4 * pc
            if best is None or net > best:
                best = net
    return best


def _upper_bound(inst):
    """Loose (unreachable) upper bound = sum of all strictly-positive values."""
    s = 0
    for row in inst["grid"]:
        for v in row:
            if v > 0:
                s += v
    return s


# ----------------------------- validation ----------------------------------
def _net_value(inst, answer):
    """Validate footprint against instance. Return net value or None if invalid."""
    if not isinstance(answer, dict):
        return None
    cells = answer.get("cells")
    if not isinstance(cells, list) or len(cells) == 0:
        return None
    H = inst["H"]
    W = inst["W"]
    grid = inst["grid"]
    pc = inst["perim_cost"]
    chosen = set()
    for cell in cells:
        if not isinstance(cell, list) or len(cell) != 2:
            return None
        r, c = cell
        if isinstance(r, bool) or isinstance(c, bool):
            return None
        if not isinstance(r, int) or not isinstance(c, int):
            return None
        if r < 0 or r >= H or c < 0 or c >= W:
            return None
        if (r, c) in chosen:
            return None                     # duplicate
        chosen.add((r, c))
    # 4-connectivity check via BFS from an arbitrary chosen cell.
    start = next(iter(chosen))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in chosen and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(chosen):
        return None                         # disconnected
    # net value = sum of chosen values - perim_cost * boundary length.
    total = 0
    perim = 0
    for (r, c) in chosen:
        total += grid[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W or (nr, nc) not in chosen:
                perim += 1
    return total - pc * perim


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        b = _baseline(inst)
        U = _upper_bound(inst)
        denom = U - b
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "H": inst["H"], "W": inst["W"],
                  "perim_cost": inst["perim_cost"],
                  "grid": [list(row) for row in inst["grid"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            c = _net_value(inst, ans)
        except Exception:
            c = None
        if c is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (c - b) / denom
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
