#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for roof-watershed-droplet-router.

Reads the instance, validates the edited heightfield strictly, then simulates the
fixed deterministic steepest-descent droplet routing on BOTH the participant's
edited heightfield and the (always-feasible) unedited original, and reports the
ratio of drained droplets vs the unedited baseline.
"""
import sys, math

DIRS = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # Up, Left, Down, Right -- fixed tie-break order
BREF_FRAC = 0.37  # Bref = this fraction of the unedited roof's own drained count


def fail(msg):
    print("INFEASIBLE: %s Ratio: 0.0" % msg)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split('\n')
    idx = 0
    R, C = map(int, toks[idx].split()); idx += 1
    S, B = map(int, toks[idx].split()); idx += 1
    H = []
    for _ in range(R):
        H.append(list(map(int, toks[idx].split())))
        idx += 1
    obstacle = []
    for _ in range(R):
        obstacle.append([ch == '#' for ch in toks[idx]])
        idx += 1
    gutter = []
    for _ in range(R):
        gutter.append([ch == 'G' for ch in toks[idx]])
        idx += 1
    return R, C, S, B, H, obstacle, gutter


def route(H, obstacle, gutter, R, C, start, maxsteps):
    pos = start
    for _ in range(maxsteps):
        r, c = pos
        if gutter[r][c]:
            return True
        best = None
        bh = None
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc] and H[nr][nc] < H[r][c]:
                if bh is None or H[nr][nc] < bh:
                    bh = H[nr][nc]
                    best = (nr, nc)
        if best is None:
            return False
        pos = best
    return False


def count_drained(H, obstacle, gutter, R, C):
    maxsteps = R * C + 5
    total = 0
    for r in range(R):
        for c in range(C):
            if obstacle[r][c]:
                continue
            total += 1
    drained = 0
    for r in range(R):
        for c in range(C):
            if obstacle[r][c]:
                continue
            if route(H, obstacle, gutter, R, C, (r, c), maxsteps):
                drained += 1
    return drained, total


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]
    R, C, S, B, H, obstacle, gutter = read_instance(inf)

    try:
        with open(outf) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != R * C:
        fail("expected exactly %d integers, got %d" % (R * C, len(raw)))

    vals = []
    for tok in raw:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token %r" % tok)
        if not math.isfinite(v):
            fail("non-finite value %r" % tok)
        if abs(v - round(v)) > 1e-6:
            fail("non-integer height %r" % tok)
        if v < -1_000_000 or v > 1_000_000:
            fail("height out of bounds %r" % tok)
        vals.append(int(round(v)))

    Hp = [vals[r * C:(r + 1) * C] for r in range(R)]

    # obstacle cells must be exactly unchanged
    for r in range(R):
        for c in range(C):
            if obstacle[r][c] and Hp[r][c] != H[r][c]:
                fail("obstacle cell (%d,%d) height changed" % (r, c))

    # edit budget: sum of |delta| over non-obstacle cells
    total_edit = 0
    for r in range(R):
        for c in range(C):
            if not obstacle[r][c]:
                total_edit += abs(Hp[r][c] - H[r][c])
    if total_edit > B:
        fail("edit budget exceeded: used %d > B=%d" % (total_edit, B))

    # max-slope constraint on the FINAL heightfield, between non-obstacle 4-neighbors
    for r in range(R):
        for c in range(C):
            if obstacle[r][c]:
                continue
            for dr, dc in ((1, 0), (0, 1)):  # each undirected edge checked once
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and not obstacle[nr][nc]:
                    if abs(Hp[r][c] - Hp[nr][nc]) > S:
                        fail("slope violated between (%d,%d) and (%d,%d)" % (r, c, nr, nc))

    F, total_cells = count_drained(Hp, obstacle, gutter, R, C)
    Bnatural, _ = count_drained(H, obstacle, gutter, R, C)  # drops the unedited roof itself drains
    # The scored reference Bref is a deliberately PESSIMISTIC fraction of that
    # natural count (not the natural count itself): the unedited roof is a
    # legitimate, always-feasible submission, but it is not treated as "the
    # target" -- only actually reconnecting basins is. This keeps genuine
    # ridge-flip work visible in the score instead of being swamped by the
    # large fraction of the roof that already drains on its own.
    Bref = max(1, round(BREF_FRAC * Bnatural))

    sc = min(1000.0, 100.0 * F / max(1e-9, Bref))
    ratio = sc / 1000.0
    print("edits=%d/%d drained=%d/%d natural=%d Bref=%d Ratio: %.6f" %
          (total_edit, B, F, total_cells, Bnatural, Bref, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
