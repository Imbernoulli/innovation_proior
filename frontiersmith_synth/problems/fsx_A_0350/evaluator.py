#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0350 -- "Selene Base: Pressurized Footprint Layout"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A lunar-habitat survey team receives a discrete site map of a landing zone:
an N x N grid of regolith tiles.  Each tile (r, c) carries an integer `net` value --
the SCIENCE + RESOURCE yield of pressurizing that tile MINUS the excavation / shielding
cost of clearing it.  Rich ore veins, ice pockets and instrument sites form a handful
of compact positive "deposits"; the bare regolith between them is net-NEGATIVE (it
still costs mass to occupy but returns nothing).

The base is a single PRESSURIZED, WALKABLE footprint: the set S of occupied tiles must
be 4-CONNECTED (every module reachable from every other through adjacent occupied
tiles -- no teleporting between disconnected domes).  Life-support caps the build at
at most K tiles.  The team MAXIMIZES the total net value of the footprint:

    value(S) = sum of net[r][c] over the occupied tiles S,
    subject to:  S non-empty, 4-connected, |S| <= K, all tiles in range & distinct.

The tension is coverage vs. connectivity vs. budget: the richest deposits sit apart,
separated by net-negative regolith moats.  Grabbing one deposit is easy; stitching two
rich deposits together means spending scarce budget on negative bridge tiles, which
only pays off if the second deposit is rich enough.  This is a budgeted maximum-weight
connected-subgraph instance skinned as an offline AtCoder-heuristic-style layout
contest, scored by a fixed deterministic formula (no wall-time, no op timing).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": N (int),                 # grid is N x N
             "k": K (int),                 # at most K occupied tiles
             "net": [[...], ...]}          # N rows of N ints (may be negative)
  stdout: ONE JSON object:
            {"cells": [[r0, c0], [r1, c1], ...]}   # the occupied footprint S
          Each [r, c] is a tile with 0 <= r, c < N.  Tiles must be pairwise DISTINCT,
          number between 1 and K, and form a single 4-connected region.

  A footprint is VALID iff `cells` is a list of 1..K pairs of integers, every
  coordinate is in range, all tiles are distinct, and the tiles are 4-connected.
  Wrong shape, out-of-range / duplicate tiles, a disconnected set, an empty set,
  more than K tiles, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance we compute two references:
    v_base = the value of the single BEST tile (max net over the grid).  A one-tile
             footprint is always connected and within budget; this overlap-free but
             coverage-free placement is the weak baseline.
    v_ub   = the sum of ALL strictly-positive net tiles (ignoring connectivity AND the
             budget K).  A loose, generally unreachable upper bound: no single
             K-budget connected footprint can gather every scattered positive tile, so
             even excellent layouts stay below 1.0 -> headroom.
    v_cand = value(S) of the candidate footprint.
  normalized with an affine anchor (single-best-tile -> 0.1, all-positive ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (v_cand - v_base) / max(1e-9, v_ub - v_base), 0, 1 )
  Reproducing the single best tile scores ~0.1; a footprint that dips net-negative
  scores below 0.1; growing a connected region across rich deposits scores higher,
  capped at 1.0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  Both references and all
validation happen in THIS parent process, so a frame-walking / introspecting candidate
learns nothing that helps it lay out the base.

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
def _build_net(seed, n, num_dep, bg):
    """Deterministic N x N integer net-value map: `num_dep` compact positive
    deposits on a net-NEGATIVE regolith floor of magnitude `bg`."""
    ni = _rng(seed)
    deps = []
    for _ in range(num_dep):
        dr = ni(0, n - 1)
        dc = ni(0, n - 1)
        amp = ni(28, 60)           # peak richness
        sig = ni(1, 2)             # spread; small -> tight compact deposit
        deps.append((dr, dc, amp, sig))
    net = [[0] * n for _ in range(n)]
    for r in range(n):
        for c in range(n):
            raw = 0
            for (dr, dc, amp, sig) in deps:
                d2 = (r - dr) * (r - dr) + (c - dc) * (c - dc)
                denom = 1 + (d2 * 100) // (2 * sig * sig)
                raw += (amp * 100) // denom
            net[r][c] = raw - bg
    return net


def _build_instances():
    """Deterministic instance family: (seed, n, k, num_dep, bg)."""
    specs = [
        (101, 14, 18, 4, 12),
        (102, 16, 22, 5, 12),
        (103, 16, 20, 5, 14),
        (104, 18, 26, 5, 12),
        (105, 18, 24, 6, 14),
        (106, 20, 30, 6, 12),
        (107, 20, 28, 7, 14),
        (108, 22, 34, 7, 12),
        # harder / larger held-out instances (more deposits than the budget can span)
        (211, 22, 30, 9, 14),
        (212, 24, 38, 8, 12),
        (213, 24, 34, 10, 14),
        (214, 26, 42, 9, 14),
    ]
    out = []
    for (seed, n, k, num_dep, bg) in specs:
        net = _build_net(seed, n, num_dep, bg)
        out.append({"name": f"site{seed}", "n": n, "k": k, "net": net})
    return out


# ----------------------------- references / scoring ------------------------
def _baseline_value(net):
    """Value of the single best tile (always a valid 1-tile footprint)."""
    return max(max(row) for row in net)


def _upper_bound(net):
    """Sum of all strictly-positive tiles (ignores connectivity and budget)."""
    return sum(v for row in net for v in row if v > 0)


def _validate(inst, answer):
    """Validate footprint against the instance. Return list of (r,c) or None."""
    if not isinstance(answer, dict):
        return None
    cells = answer.get("cells")
    if not isinstance(cells, list):
        return None
    n = inst["n"]; k = inst["k"]
    if len(cells) < 1 or len(cells) > k:
        return None
    seen = set()
    out = []
    for pair in cells:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        r, c = pair[0], pair[1]
        if isinstance(r, bool) or isinstance(c, bool):
            return None
        if not isinstance(r, int) or not isinstance(c, int):
            return None
        if r < 0 or r >= n or c < 0 or c >= n:
            return None
        if (r, c) in seen:
            return None
        seen.add((r, c))
        out.append((r, c))
    # 4-connectivity check via BFS over the selected set
    start = out[0]
    stack = [start]
    reached = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in seen and nb not in reached:
                reached.add(nb)
                stack.append(nb)
    if len(reached) != len(seen):
        return None
    return out


def _footprint_value(net, cells):
    return sum(net[r][c] for (r, c) in cells)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        net = inst["net"]; n = inst["n"]; k = inst["k"]
        v_base = _baseline_value(net)
        v_ub = _upper_bound(net)
        denom = v_ub - v_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": n, "k": k,
                  "net": [list(row) for row in net]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            cells = _validate(inst, ans)
        except Exception:
            cells = None
        if cells is None:
            vec.append(0.0)
            continue
        v_cand = _footprint_value(net, cells)
        r = 0.1 + 0.9 * (v_cand - v_base) / denom
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
