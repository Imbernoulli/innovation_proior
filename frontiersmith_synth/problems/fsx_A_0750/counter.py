import sys

# Format D checker -- pocket-clearing straight-line stamp program.
#   1) Parse the pocket grid + tool catalog + tool-change cost C from <in>.
#   2) Parse the participant's stamp PROGRAM from <out>:
#         N
#         N lines:  s r c     (tool size s, top-left row r, top-left col c)
#   3) FEASIBILITY gate: every stamp must use a catalog size, lie fully in bounds, and its
#      s x s footprint must be entirely pocket cells; the union of all stamps must cover
#      EVERY pocket cell. Any violation -> Ratio: 0.0.
#   4) Objective (minimize) F = N + C * (# tool changes), where a tool change is counted at
#      every adjacent pair of PROGRAM-ORDER stamps whose sizes differ (program order = the
#      order the stamps are listed in the output, not sorted).
#      Baseline B = number of pocket cells (the naive all-size-1, single-block program).
#      Ratio = min(1, 0.1 * B / F).

MAXN = 200000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    with open(sys.argv[1]) as f:
        in_lines = f.read().split("\n")
    with open(sys.argv[2]) as f:
        out_tokens = f.read().split()

    # ---- parse instance ----
    try:
        header = in_lines[0].split()
        H, W, K, C = int(header[0]), int(header[1]), int(header[2]), int(header[3])
        catalog = [int(x) for x in in_lines[1].split()]
        if len(catalog) != K:
            raise ValueError("catalog length mismatch")
        grid = []
        for r in range(H):
            row = in_lines[2 + r]
            if len(row) != W:
                raise ValueError("row width mismatch")
            grid.append(row)
    except Exception:
        fail("bad instance (harness bug, not participant fault)")

    catalog_set = set(catalog)
    total_pocket = sum(row.count('#') for row in grid)
    if total_pocket == 0:
        fail("degenerate empty pocket")

    # prefix sum over pocket cells, for O(1) box-fully-pocket checks
    pre = [[0] * (W + 1) for _ in range(H + 1)]
    for r in range(H):
        rowsum = 0
        for c in range(W):
            rowsum += 1 if grid[r][c] == '#' else 0
            pre[r + 1][c + 1] = pre[r][c + 1] + rowsum

    def box_sum(r, c, s):
        return pre[r + s][c + s] - pre[r][c + s] - pre[r + s][c] + pre[r][c]

    # ---- parse participant output ----
    it = iter(out_tokens)

    def next_int():
        tok = next(it)
        # reject non-finite / non-integer tokens explicitly (nan, inf, 1.5, 1e3, ...)
        if not (tok.lstrip('+-').isdigit()):
            raise ValueError("not an integer token: %r" % tok)
        return int(tok)

    try:
        N = next_int()
    except Exception:
        fail("bad or missing N")

    if N < 0 or N > MAXN:
        fail("N out of range")
    if N == 0:
        fail("no stamps but pocket is nonempty")

    program = []
    try:
        for _ in range(N):
            s = next_int()
            r = next_int()
            c = next_int()
            program.append((s, r, c))
        # no trailing garbage tokens allowed
        next(it)
        fail("trailing tokens after program")
    except StopIteration:
        pass
    except ValueError as e:
        fail(str(e))
    except Exception:
        fail("bad stamp token")

    if len(program) != N:
        fail("stamp count mismatch")

    # ---- feasibility ----
    covered = [[False] * W for _ in range(H)]
    for (s, r, c) in program:
        if s not in catalog_set:
            fail("tool size %d not in catalog" % s)
        if s < 1 or r < 0 or c < 0 or r + s > H or c + s > W:
            fail("stamp out of bounds (s=%d r=%d c=%d)" % (s, r, c))
        if box_sum(r, c, s) != s * s:
            fail("stamp not fully inside pocket (s=%d r=%d c=%d)" % (s, r, c))
        for rr in range(r, r + s):
            row = covered[rr]
            for cc in range(c, c + s):
                row[cc] = True

    for r in range(H):
        grow = grid[r]
        crow = covered[r]
        for c in range(W):
            if grow[c] == '#' and not crow[c]:
                fail("pocket cell (%d,%d) left uncovered" % (r, c))

    # ---- objective ----
    changes = sum(1 for i in range(1, N) if program[i][0] != program[i - 1][0])
    F = N + C * changes
    B = total_pocket
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("N=%d changes=%d F=%d B=%d Ratio: %.6f" % (N, changes, F, B, ratio))


if __name__ == "__main__":
    main()
