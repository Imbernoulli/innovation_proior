import sys

# ---------------------------------------------------------------------------
# Deterministic checker for "Finding the small colorful triangle" (Sperner
# coloring, query-budget path-following).
#
# Input  (<in>):  "N D" then (N+1)(N+2)/2 lines "x y c" giving the FULL true
#                 3-coloring of the barycentric grid of side N (this is
#                 whitebox: the participant sees every color).
# Output (<out>): a certificate of the following shape (see statement.md):
#     ANSWER x1 y1 x2 y2 x3 y3
#     PATH m
#     x0 y0 x1 y1 x2 y2        (T_0 .. T_{m-1}, one line per triangle)
#     ...
#     EXTRA k                  (optional; defaults to k=0 if omitted)
#     x y                      (k extra "looked-at" grid points)
#
# Feasibility (all mandatory for ANY nonzero score):
#   - ANSWER is a genuine small triangle of the grid and is panchromatic
#     (uses true colors from the input).
#   - PATH is a COMPLETE, structurally valid door-to-door certificate: T_0 has
#     an edge on the z=0 boundary whose two endpoints are colored {0,1}; each
#     consecutive pair T_k, T_{k+1} shares exactly one edge (two vertices) and
#     that shared edge is a {0,1}-colored "door"; no triangle repeats; and the
#     LAST triangle of PATH is exactly ANSWER.  (By Sperner's parity lemma this
#     certificate -- if it exists at all -- is UNIQUE, so it cannot be forged
#     shorter than the true walk.)
#
# Objective (MAX): query efficiency.  Let OPT = the size (distinct vertices)
# of the TRUE minimal certificate, computed independently by this checker via
# the same door-walk simulation.  Let USED = the number of distinct grid
# points appearing in the submission's PATH plus its EXTRA section (points a
# submission "looked at" beyond what the mandatory certificate needed).
# USED >= OPT always for any valid submission (PATH alone already costs OPT).
#     Ratio = min(1.0, 0.85 * OPT / USED)
# so the unique optimal certificate (USED == OPT, no EXTRA) scores 0.85; every
# unnecessary looked-at vertex reported in EXTRA lowers the score.
# ---------------------------------------------------------------------------

X = 0.85


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def up_tri(i, j, N):
    if i + j > N - 1:
        return None
    return ((i, j), (i + 1, j), (i, j + 1))


def down_tri(i, j, N):
    if i + j > N - 2:
        return None
    return ((i + 1, j), (i, j + 1), (i + 1, j + 1))


def find_start_i(N, COL):
    prev = None
    for x in range(N, -1, -1):
        y = N - x
        c = COL[(x, y)]
        if prev is not None and prev != c and {prev, c} == {0, 1}:
            return x
        prev = c
    return None


def checker_opt(N, COL):
    """Checker's own minimal-certificate size, via the forced door walk."""
    i0 = find_start_i(N, COL)
    if i0 is None:
        return None
    j0 = N - i0 - 1
    cur = ((i0, j0), (i0 + 1, j0), (i0, j0 + 1))
    visited = set(cur)
    entry_edge = frozenset([cur[1], cur[2]])
    max_steps = (N + 2) * (N + 2) * 2 + 10
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            return None
        cols = {COL[v] for v in cur}
        if cols == {0, 1, 2}:
            return len(visited)
        edges = [frozenset([cur[a], cur[b]]) for a, b in [(0, 1), (1, 2), (0, 2)]]
        doors = []
        for e in edges:
            pts = list(e)
            if {COL[pts[0]], COL[pts[1]]} == {0, 1}:
                doors.append(e)
        others = [e for e in doors if e != entry_edge]
        if not others:
            return None
        exit_edge = others[0]
        third = [v for v in cur if v not in exit_edge][0]
        pA, pB = list(exit_edge)
        newv = (pA[0] + pB[0] - third[0], pA[1] + pB[1] - third[1])
        if newv[0] < 0 or newv[1] < 0 or newv[0] + newv[1] > N:
            return None
        cur = (pA, pB, newv)
        visited |= set(cur)
        entry_edge = exit_edge


def is_valid_triangle(N, verts):
    """verts: 3 distinct (x,y) integer pairs forming a genuine UP or DOWN
    small triangle of the grid (0<=x,0<=y,x+y<=N)."""
    if len(verts) != 3:
        return False
    for (x, y) in verts:
        if x < 0 or y < 0 or x + y > N:
            return False
    if len(set(verts)) != 3:
        return False
    i0 = min(v[0] for v in verts)
    j0 = min(v[1] for v in verts)
    rel = sorted((v[0] - i0, v[1] - j0) for v in verts)
    return rel == [(0, 0), (0, 1), (1, 0)] or rel == [(0, 1), (1, 0), (1, 1)]


def main():
    # ---- read the instance (trusted: produced by gen.py) ----
    try:
        itoks = open(sys.argv[1]).read().split()
        N = int(itoks[0]); D = int(itoks[1])
        COL = {}
        p = 2
        while p + 2 < len(itoks) + 1 and p < len(itoks):
            x = int(itoks[p]); y = int(itoks[p + 1]); c = int(itoks[p + 2])
            COL[(x, y)] = c
            p += 3
    except Exception:
        fail("bad input")

    V_total = (N + 1) * (N + 2) // 2
    if len(COL) != V_total:
        fail("bad input grid size")

    OPT = checker_opt(N, COL)
    if OPT is None:
        fail("internal: no certificate exists for this instance")

    # ---- bounded, defensive parsing of the participant output ----
    MAX_TOKENS = 20 * V_total + 200000
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")
    if len(raw) > 30_000_000:
        fail("output too large")
    toks = raw.split()
    if len(toks) > MAX_TOKENS:
        fail("too many tokens")
    if not toks:
        fail("empty output")

    pos = 0

    def next_tok():
        nonlocal pos
        if pos >= len(toks):
            raise IndexError("truncated output")
        t = toks[pos]
        pos += 1
        return t

    def next_int():
        t = next_tok()
        v = int(t)  # raises ValueError on "nan"/"inf"/garbage -> caught below
        return v

    try:
        if next_tok() != "ANSWER":
            fail("expected literal token ANSWER")
        ax = [next_int() for _ in range(6)]
        answer = ((ax[0], ax[1]), (ax[2], ax[3]), (ax[4], ax[5]))
        if not is_valid_triangle(N, answer):
            fail("ANSWER is not a valid small triangle")
        for v in answer:
            if v not in COL:
                fail("ANSWER vertex out of grid")
        ans_cols = {COL[v] for v in answer}
        if ans_cols != {0, 1, 2}:
            fail("ANSWER triangle is not panchromatic")

        if next_tok() != "PATH":
            fail("expected literal token PATH")
        m = next_int()
        MAX_PATH = 20 * (N + 5)
        if m < 1 or m > MAX_PATH:
            fail("PATH length out of bounds")

        path = []
        for _ in range(m):
            tv = [next_int() for _ in range(6)]
            tri = ((tv[0], tv[1]), (tv[2], tv[3]), (tv[4], tv[5]))
            if not is_valid_triangle(N, tri):
                fail("PATH contains an invalid triangle")
            for v in tri:
                if v not in COL:
                    fail("PATH vertex out of grid")
            path.append(tri)

        # T_0 must have a genuine z=0-boundary {0,1} door edge.
        t0 = path[0]
        boundary_ok = False
        for a in range(3):
            for b in range(a + 1, 3):
                va, vb = t0[a], t0[b]
                za = N - va[0] - va[1]; zb = N - vb[0] - vb[1]
                if za == 0 and zb == 0 and {COL[va], COL[vb]} == {0, 1}:
                    boundary_ok = True
        if not boundary_ok:
            fail("PATH does not start at a genuine boundary door")

        seen_tris = set()
        seen_tris.add(frozenset(t0))
        entry_edge = None
        for k in range(m - 1):
            Tk, Tk1 = path[k], path[k + 1]
            shared = set(Tk) & set(Tk1)
            if len(shared) != 2:
                fail("PATH step %d not edge-adjacent" % k)
            u, v = list(shared)
            if {COL[u], COL[v]} != {0, 1}:
                fail("PATH step %d does not cross a {0,1} door" % k)
            key = frozenset(Tk1)
            if key in seen_tris:
                fail("PATH revisits a triangle")
            seen_tris.add(key)

        if frozenset(path[-1]) != frozenset(answer):
            fail("PATH does not terminate at ANSWER")

        used = set()
        for tri in path:
            used |= set(tri)

        # ---- optional EXTRA section ----
        if pos < len(toks) and toks[pos] == "EXTRA":
            next_tok()
            k = next_int()
            MAX_EXTRA = 20 * V_total + 1000
            if k < 0 or k > MAX_EXTRA:
                fail("EXTRA count out of bounds")
            for _ in range(k):
                x = next_int(); y = next_int()
                if x < 0 or y < 0 or x + y > N:
                    fail("EXTRA point out of grid")
                used.add((x, y))
    except IndexError:
        fail("truncated output")
    except (ValueError, KeyError):
        fail("parse error / non-finite or out-of-range token")

    used_n = len(used)
    if used_n < OPT:
        # cannot happen for a genuinely valid PATH (it alone contributes OPT
        # distinct vertices), but guard anyway.
        used_n = OPT

    ratio = min(1.0, X * OPT / max(1, used_n))
    print("OPT=%d USED=%d Ratio: %.6f" % (OPT, used_n, ratio))


if __name__ == "__main__":
    main()
