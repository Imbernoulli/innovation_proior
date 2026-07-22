import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp_tokens = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    # ---- parse instance ----
    try:
        it = iter(inp_tokens)
        R = int(next(it)); C = int(next(it)); base = int(next(it))
        K = int(next(it))
        counters = []  # (r, c, d, w, scope)
        for _ in range(K):
            r = int(next(it)); c = int(next(it)); d = int(next(it)); w = int(next(it))
            scope = next(it)
            counters.append((r, c, d, w, scope))
        H = int(next(it))
        hints = []  # (r, c, v)
        for _ in range(H):
            r = int(next(it)); c = int(next(it)); v = int(next(it))
            hints.append((r, c, v))
    except Exception:
        fail("bad input")

    if R <= 0 or C <= 0 or base <= 0 or R % 2 != 0:
        fail("bad instance geometry")

    hint_map = {}
    for (r, c, v) in hints:
        hint_map[(r, c)] = v

    def partner(r):
        return (r + R // 2) % R

    def scope_cells(r, scope):
        if scope == "ROW":
            return [(r, cc) for cc in range(C)]
        elif scope == "PAIR":
            rp = partner(r)
            cells = [(r, cc) for cc in range(C)]
            cells += [(rp, cc) for cc in range(C)]
            return cells
        else:
            return None

    for (r, c, d, w, scope) in counters:
        if not (0 <= r < R and 0 <= c < C and 0 <= d < base and w >= 1):
            fail("bad counter spec")
        if scope not in ("ROW", "PAIR"):
            fail("bad scope token")

    # ---- parse participant output: exactly R*C integer tokens, R lines of C each ----
    try:
        out_lines_raw = open(sys.argv[2]).read().strip("\n").split("\n")
    except Exception:
        fail("cannot re-read output")
    # be lenient about trailing blank lines but strict about the row/col shape otherwise
    out_lines = [ln for ln in out_lines_raw if ln.strip() != ""]
    if len(out_lines) != R:
        fail("expected %d output lines, got %d" % (R, len(out_lines)))

    grid = []
    for r in range(R):
        toks = out_lines[r].split()
        if len(toks) != C:
            fail("row %d: expected %d tokens, got %d" % (r, C, len(toks)))
        row_vals = []
        for tok in toks:
            try:
                v = int(tok)
            except Exception:
                fail("row %d: non-integer token %r" % (r, tok))
            if not (0 <= v < base):
                fail("row %d: digit %d out of range [0,%d)" % (r, v, base))
            row_vals.append(v)
        grid.append(row_vals)

    # ---- feasibility: seed hints must be respected exactly ----
    for (r, c, v) in hints:
        if grid[r][c] != v:
            fail("hint violated at (%d,%d): expected %d, got %d" % (r, c, v, grid[r][c]))

    def weighted_satisfied(g):
        total = 0
        for (r, c, d, w, scope) in counters:
            cells = scope_cells(r, scope)
            actual = 0
            for (rr, cc) in cells:
                if g[rr][cc] == d:
                    actual += 1
            if g[r][c] == actual:
                total += w
        return total

    F = weighted_satisfied(grid)

    # ---- internal baseline B: checker's own trivial tableau T0 ----
    # T0: every non-hint cell gets (column index) mod base; hints keep their fixed value.
    T0 = [[c % base for c in range(C)] for r in range(R)]
    for (r, c, v) in hints:
        T0[r][c] = v
    B = weighted_satisfied(T0)
    B = max(1, B)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
