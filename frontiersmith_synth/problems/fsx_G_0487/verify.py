import sys

# Deterministic scorer for the corner-free grid scheduling problem.
#
#   python3 verify.py <in> <out> <ans>   (ans is an ignored placeholder)
#
# A "conflict corner" is three reserved cells forming an axis-aligned right-isosceles L:
#       (r, c), (r + d, c), (r, c + d)   with d >= 1.
# A valid schedule is a set of distinct, in-range, non-blocked cells containing NO corner.
# Objective (maximize): number of reserved cells.
#
# Internal baseline B = the largest single time slot (row) that can be fully reserved after
# removing blocked cells. A single row is ALWAYS corner-free (a corner needs a cell in a
# different row), so B is a legitimate trivial feasible construction.

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    # ---- parse instance ----
    try:
        it = iter(inp)
        N = int(next(it))
        nb = int(next(it))
        blocked = set()
        for _ in range(nb):
            r = int(next(it)); c = int(next(it))
            blocked.add((r, c))
    except Exception:
        fail("bad instance")

    if N <= 0 or nb < 0:
        fail("bad instance params")

    # ---- internal baseline B: best fully-reservable single row ----
    B = 0
    for r in range(N):
        cnt = sum(1 for c in range(N) if (r, c) not in blocked)
        if cnt > B:
            B = cnt
    B = max(1, B)

    # ---- parse participant output (integers only; nan/inf tokens fail int()) ----
    if len(out_raw) == 0:
        fail("empty output")
    try:
        M = int(out_raw[0])
    except Exception:
        fail("bad count token")
    if M < 0:
        fail("negative count")
    # reject non-finite / non-integer coordinate tokens explicitly
    for tok in out_raw:
        low = tok.lower()
        if ("nan" in low) or ("inf" in low) or ("." in tok) or ("e" in low):
            fail("non-integer / non-finite token %r" % tok)

    body = out_raw[1:]
    if len(body) != 2 * M:
        fail("expected %d coordinate tokens, got %d" % (2 * M, len(body)))

    cells = []
    try:
        for i in range(M):
            r = int(body[2 * i]); c = int(body[2 * i + 1])
            cells.append((r, c))
    except Exception:
        fail("bad coordinate")

    # ---- feasibility: range, blocked, duplicates ----
    S = set()
    for (r, c) in cells:
        if not (0 <= r < N and 0 <= c < N):
            fail("cell out of range (%d,%d)" % (r, c))
        if (r, c) in blocked:
            fail("blocked cell reserved (%d,%d)" % (r, c))
        if (r, c) in S:
            fail("duplicate cell (%d,%d)" % (r, c))
        S.add((r, c))

    # ---- corner-free check (O(sum of row/col degrees)) ----
    rowmap = {}   # row -> set of cols
    colmap = {}   # col -> set of rows
    for (r, c) in S:
        rowmap.setdefault(r, set()).add(c)
        colmap.setdefault(c, set()).add(r)

    for (r, c) in S:
        # candidate horizontal offsets d>0 with (r, c+d) in S
        right = rowmap[r]
        # candidate vertical   offsets d>0 with (r+d, c) in S
        up = colmap[c]
        # corner iff exists d>=1 with (r, c+d) and (r+d, c) both present
        # iterate over the smaller degree set
        if len(right) <= len(up):
            for cc in right:
                d = cc - c
                if d >= 1 and (r + d) in up:
                    fail("conflict corner at (%d,%d) d=%d" % (r, c, d))
        else:
            for rr in up:
                d = rr - r
                if d >= 1 and (c + d) in right:
                    fail("conflict corner at (%d,%d) d=%d" % (r, c, d))

    F = len(S)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
