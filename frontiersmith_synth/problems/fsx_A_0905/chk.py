#!/usr/bin/env python3
"""
chk.py -- checker/scorer for fsx_A_0905 (mosaic-gradient-tessera).

Usage: python3 chk.py <in> <out> <ans>   (ans is unused; scorer problem)

Prints a line containing `Ratio: <float in [0,1]>` and ALWAYS exits 0
(the harness trusts a python checker's score only if it exits 0).

Objective (see statement.md for the full spec):
  fidelity   = sum over cells of (vrange - |v[color] - target(radius)|)
  dispersion = ring-transition count / max-possible-ring-transitions,
               where a "ring" is the set of cells sharing (nearest-
               integer) Euclidean distance to the grid center, and a
               transition is a pair of angularly-adjacent same-ring cells
               (circular) that hold DIFFERENT colors.
  F = fidelity * (1 + BONUS * dispersion)
B is the SAME formula applied to the checker's own diagonal-residue
"trivial" construction color(i,j) = (i+j) mod c (deterministic, always
run-cap feasible for any K>=1, completely target-blind).
"""
import math
import sys

BONUS = 140.0
# The diagonal-residue baseline is a genuinely competent target-blind
# construction (good dispersion "for free"), so the raw achievable range
# above it is modest -- calibrating trivial to F(baseline)/CAL_DIV instead
# of F(baseline) keeps trivial in a sane low band (~0.3, still well inside
# the "not close to optimal" zone) while opening real headroom between the
# greedy/strong reference tiers and the sc=1.0 cap for the RL policy to
# still improve toward.
CAL_DIV = 3.4


def fail(reason, F=0.0, B=1.0):
    print(f"WA {reason} F={F} B={B} Ratio: 0.000000")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it)); c = int(next(it)); K = int(next(it))
    v = [int(next(it)) for _ in range(c)]
    cnt = [int(next(it)) for _ in range(c)]
    tcenter = int(next(it)); tedge = int(next(it))
    return n, c, K, v, cnt, tcenter, tedge


def target_fn(n, tcenter, tedge):
    cy = cx = (n - 1) / 2.0
    rmax = math.hypot(cx, cy) if n > 1 else 1.0
    if rmax <= 0:
        rmax = 1.0

    def t(i, j):
        r = math.hypot(i - cy, j - cx)
        return tcenter + (tedge - tcenter) * (r / rmax)
    return t


def diagonal_grid(n, c):
    """The checker's own trivial, target-blind, always-cap-safe construction:
    color(i,j) = (i+j) mod c. Any two orthogonally-adjacent cells differ by
    exactly 1 (mod c) in this index, so no run ever exceeds length 1 -- safe
    for every K>=1. Deliberately identical to solutions/trivial.py."""
    return [[(i + j) % c for j in range(n)] for i in range(n)]


def ring_groups(n):
    """Bucket cells by rounded Euclidean distance to center ('ring'); within
    a ring, order by angle (deterministic circular order)."""
    cy = cx = (n - 1) / 2.0
    rings = {}
    for i in range(n):
        for j in range(n):
            r = math.hypot(i - cy, j - cx)
            ang = math.atan2(i - cy, j - cx)
            b = int(r + 0.5)
            rings.setdefault(b, []).append((ang, i, j))
    for b in rings:
        rings[b].sort()
    return rings


def ring_transition_stats(rings):
    """Return (max_possible_transitions) = sum of ring sizes for rings with
    >=2 members (a ring of size m can have at most m circular transitions)."""
    return sum(len(lst) for lst in rings.values() if len(lst) >= 2)


def score_grid(grid, n, v, t, rings, max_rt):
    vrange = max(1, v[-1] - v[0])
    fid = 0.0
    for i in range(n):
        row = grid[i]
        for j in range(n):
            err = abs(v[row[j]] - t(i, j))
            fid += (vrange - err)
    rt = 0
    for b, lst in rings.items():
        m = len(lst)
        if m < 2:
            continue
        colors = [grid[i][j] for _, i, j in lst]
        rt += sum(1 for x in range(m) if colors[x] != colors[(x + 1) % m])
    normdiv = (rt / max_rt) if max_rt > 0 else 0.0
    return fid * (1.0 + BONUS * normdiv)


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, ouf = sys.argv[1], sys.argv[2]
    try:
        n, c, K, v, cnt, tcenter, tedge = read_instance(inf)
    except Exception as e:
        fail(f"bad-input:{e}")

    try:
        with open(ouf) as f:
            toks = f.read().split()
    except Exception as e:
        fail(f"no-output:{e}")

    if len(toks) != n * n:
        fail(f"token-count {len(toks)} != {n*n}")

    grid = [[0] * n for _ in range(n)]
    seen_cnt = [0] * c
    idx = 0
    for i in range(n):
        for j in range(n):
            tok = toks[idx]; idx += 1
            # strict integer parse: a single optional leading sign followed by
            # only digits -- rejects nan/inf/garbage/huge-non-int tokens AND
            # malformed multi-sign tokens like "--1" or "+-1" (isdigit() alone
            # is fooled by those since str.lstrip("+-") strips ALL of them).
            body = tok[1:] if tok and tok[0] in "+-" else tok
            if not body.isdigit():
                fail(f"non-integer token '{tok}' at ({i},{j})")
            col1 = int(tok)
            if col1 < 1 or col1 > c:
                fail(f"color {col1} out of range [1,{c}] at ({i},{j})")
            col0 = col1 - 1
            grid[i][j] = col0
            seen_cnt[col0] += 1

    if seen_cnt != cnt:
        fail(f"multiset mismatch seen={seen_cnt} want={cnt}")

    # run-length cap: no orthogonal run of same color exceeds K
    for i in range(n):
        run_c = grid[i][0]; run_l = 1
        for j in range(1, n):
            if grid[i][j] == run_c:
                run_l += 1
                if run_l > K:
                    fail(f"row {i} run of color {run_c+1} exceeds K={K} at col {j}")
            else:
                run_c = grid[i][j]; run_l = 1
    for j in range(n):
        run_c = grid[0][j]; run_l = 1
        for i in range(1, n):
            if grid[i][j] == run_c:
                run_l += 1
                if run_l > K:
                    fail(f"col {j} run of color {run_c+1} exceeds K={K} at row {i}")
            else:
                run_c = grid[i][j]; run_l = 1

    t = target_fn(n, tcenter, tedge)
    rings = ring_groups(n)
    max_rt = ring_transition_stats(rings)

    F = score_grid(grid, n, v, t, rings, max_rt)

    base_grid = diagonal_grid(n, c)
    B = score_grid(base_grid, n, v, t, rings, max_rt) / CAL_DIV
    if B <= 0:
        B = 1.0

    sc = min(1000.0, 100.0 * F / max(1.0, B))
    ratio = sc / 1000.0
    print(f"OK F={F:.4f} B={B:.4f} Ratio: {ratio:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        # Defensive catch-all: an unanticipated crash must still be a clean
        # WA (exit 0, Ratio 0), never a nonzero exit with no Ratio line.
        fail(f"checker-exception:{e}")
