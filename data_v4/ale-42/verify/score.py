#!/usr/bin/env python3
"""Deterministic local scorer for "Interactive Adaptive Probing" (ale-42).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--raw]

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

THE PROBLEM (see context.md "Evaluation settings").
A hidden H x W grid carries a non-negative integer REWARD field r(x, y). A cell
is HOT iff r(x, y) >= thr. A probing agent has a budget of Q probes; a probe at
(x, y) OBSERVES every cell within Chebyshev radius `rad` of (x, y). The agent
must end with a REPORT: a set of cells it declares hot. The score rewards
reporting the true hot mass and penalizes false positives -- but you may only
report cells you actually observed (probed near), and you may not exceed Q
probes.

INSTANCE (read from INSTANCE_FILE):
    H W Q rad sigma thr penalty
    then H rows of W integers, r[x][y].

SOLUTION (read from SOLUTION_FILE), whitespace-separated tokens:
    P                    (number of probes; integer)
    px_1 py_1            (P probe cells, row then column)
    ...
    M                    (number of reported cells; integer)
    rx_1 ry_1            (M reported cells, row then column)
    ...

FEASIBILITY (any violation -> score 0):
  * the token stream parses exactly: P >= 0, then 2*P probe ints, then M >= 0,
    then 2*M report ints, and NOTHING left over;
  * P <= Q  (the probe budget; sigma is the agent's measurement noise, used only
    to motivate the difficulty -- the scorer scores against the TRUE field);
  * every probe cell and every report cell is inside the grid
    (0 <= x < H, 0 <= y < W);
  * the M reported cells are pairwise DISTINCT;
  * every reported cell is OBSERVED: it lies within Chebyshev radius `rad` of at
    least one probe cell. Reporting a cell you never looked at is illegal.

OBJECTIVE (of a feasible report): with HOT(c) := (r(c) >= thr),
    value = sum over reported c of r(c)
            - penalty * (number of reported c that are NOT hot, i.e. r(c) < thr)
A report that names only true hot cells collects their full reward with no
penalty; naming an observed-but-cold cell costs `penalty`. value may be 0 (empty
report) but is never used as a divisor.

SCORE (higher better), normalized against a deterministic UNIFORM-GRID probing
baseline the scorer recomputes itself: lay Q probes on a regular lattice
covering the grid as evenly as possible, observe their windows, and report every
OBSERVED cell whose true reward is >= thr (the omniscient-within-window report,
which is the best any reporter can do given that fixed coverage). Then
    score = round(1_000_000 * solver_value / max(1, baseline_value))
INFEASIBLE -> 0. With --raw the raw integer `value` is printed instead.
"""
import sys


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    H = int(next(it)); W = int(next(it)); Q = int(next(it))
    rad = int(next(it)); sigma = float(next(it))
    thr = int(next(it)); penalty = int(next(it))
    grid = [[0] * W for _ in range(H)]
    for x in range(H):
        for y in range(W):
            grid[x][y] = int(next(it))
    return H, W, Q, rad, sigma, thr, penalty, grid


def read_solution(path, H, W, Q, rad, grid):
    """Parse + fully validate. Return (probes, report) or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    pos = 0
    n = len(toks)

    def take_int():
        nonlocal pos
        if pos >= n:
            raise ValueError
        v = int(toks[pos]); pos += 1
        return v

    try:
        P = take_int()
        if P < 0 or P > Q:
            return None
        probes = []
        for _ in range(P):
            x = take_int(); y = take_int()
            if x < 0 or x >= H or y < 0 or y >= W:
                return None
            probes.append((x, y))
        M = take_int()
        if M < 0:
            return None
        report = []
        for _ in range(M):
            x = take_int(); y = take_int()
            if x < 0 or x >= H or y < 0 or y >= W:
                return None
            report.append((x, y))
    except ValueError:
        return None
    if pos != n:                     # trailing junk -> malformed
        return None

    # reported cells must be pairwise distinct.
    seen = set()
    for c in report:
        if c in seen:
            return None
        seen.add(c)

    # every reported cell must be OBSERVED: within Chebyshev rad of some probe.
    if report:
        observed = set()
        for (px, py) in probes:
            x0 = max(0, px - rad); x1 = min(H - 1, px + rad)
            y0 = max(0, py - rad); y1 = min(W - 1, py + rad)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    observed.add((x, y))
        for c in report:
            if c not in observed:
                return None
    return probes, report


# ------------------------------------------------------------------------- value
def report_value(grid, thr, penalty, report):
    """sum r(c) over reported cells, minus penalty per reported NON-hot cell."""
    val = 0
    for (x, y) in report:
        r = grid[x][y]
        if r >= thr:
            val += r
        else:
            val += r - penalty
    return val


# ------------------------------------------------ baseline: uniform-grid probing
def baseline_value(H, W, Q, rad, thr, penalty, grid):
    """Lay Q probes on a regular lattice; report every observed hot cell.

    Choose a grid of rows x cols (rows * cols <= Q) as close to square as
    possible, place probes at the centers of evenly spaced cells, observe their
    (2 rad + 1)^2 windows, and report exactly the observed cells whose true
    reward is >= thr (no false positives -- this is the best report for that
    fixed coverage). The collected value is the normalizer.
    """
    if Q <= 0:
        return 1
    # pick rows x cols ~ aspect-matched, product <= Q, maximized.
    best = (1, 1)
    best_prod = 1
    for rows in range(1, Q + 1):
        cols = Q // rows
        if cols < 1:
            continue
        prod = rows * cols
        # prefer more probes used; tie-break toward squarer aspect vs grid shape.
        if prod > best_prod or (
            prod == best_prod
            and abs(rows / max(1, cols) - H / max(1, W))
            < abs(best[0] / max(1, best[1]) - H / max(1, W))
        ):
            best = (rows, cols)
            best_prod = prod
    rows, cols = best

    probes = []
    for i in range(rows):
        # center of the i-th of `rows` equal horizontal bands
        px = int((i + 0.5) * H / rows)
        px = min(H - 1, max(0, px))
        for j in range(cols):
            py = int((j + 0.5) * W / cols)
            py = min(W - 1, max(0, py))
            probes.append((px, py))

    observed = set()
    for (px, py) in probes:
        x0 = max(0, px - rad); x1 = min(H - 1, px + rad)
        y0 = max(0, py - rad); y1 = min(W - 1, py + rad)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                observed.add((x, y))

    report = [(x, y) for (x, y) in observed if grid[x][y] >= thr]
    val = report_value(grid, thr, penalty, report)
    return val


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--raw]\n")
        sys.exit(1)
    raw_mode = "--raw" in sys.argv[3:]
    H, W, Q, rad, sigma, thr, penalty, grid = read_instance(sys.argv[1])

    parsed = read_solution(sys.argv[2], H, W, Q, rad, grid)
    if parsed is None:
        print(0)
        return
    probes, report = parsed

    val = report_value(grid, thr, penalty, report)
    if raw_mode:
        print(val)
        return

    base = baseline_value(H, W, Q, rad, thr, penalty, grid)
    # value can be 0 or negative (a bad report); clamp the credited value at 0 so
    # a deliberately-empty report scores 0 rather than negative.
    credited = max(0, val)
    score = int(round(1_000_000.0 * credited / max(1, base)))
    print(score)


if __name__ == "__main__":
    main()
