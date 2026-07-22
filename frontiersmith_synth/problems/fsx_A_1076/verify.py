import sys
import math

# ---------------------------------------------------------------------------
# Group of the fold arrangement (subgroup of the dihedral group D4 of the
# square grid, acting on cell indices (r,c) with r,c in [0,N)).
# ---------------------------------------------------------------------------
def group_elements(N, t):
    def idf(r, c):
        return (r, c)

    def flipH(r, c):       # mirror across the vertical center line (reflect columns)
        return (r, N - 1 - c)

    def flipV(r, c):       # mirror across the horizontal center line (reflect rows)
        return (N - 1 - r, c)

    def rot180(r, c):
        return (N - 1 - r, N - 1 - c)

    def flipD(r, c):       # main diagonal
        return (c, r)

    def flipAD(r, c):      # anti diagonal
        return (N - 1 - c, N - 1 - r)

    def rot90(r, c):
        return (c, N - 1 - r)

    def rot270(r, c):
        return (N - 1 - c, r)

    if t == 2:
        return [idf, flipH]
    if t == 4:
        return [idf, flipH, flipV, rot180]
    if t == 8:
        return [idf, flipH, flipV, rot180, flipD, flipAD, rot90, rot270]
    raise ValueError("bad fold type")


def orbit(r, c, G):
    return set(g(r, c) for g in G)


# ---------------------------------------------------------------------------
# Canonical shape signature: normalize a connected cell-set up to translation
# and the 8 planar isometries (exact integer arithmetic).
# ---------------------------------------------------------------------------
_ISOS = [
    lambda r, c: (r, c), lambda r, c: (r, -c),
    lambda r, c: (-r, c), lambda r, c: (-r, -c),
    lambda r, c: (c, r), lambda r, c: (c, -r),
    lambda r, c: (-c, r), lambda r, c: (-c, -r),
]


def canonical_shape(cells):
    best = None
    for iso in _ISOS:
        pts = [iso(r, c) for (r, c) in cells]
        minr = min(p[0] for p in pts)
        minc = min(p[1] for p in pts)
        norm = tuple(sorted((p[0] - minr, p[1] - minc) for p in pts))
        if best is None or norm < best:
            best = norm
    return best


def neighbors4(r, c):
    yield r + 1, c
    yield r - 1, c
    yield r, c + 1
    yield r, c - 1


def connected_components(cellset):
    remaining = set(cellset)
    comps = []
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        remaining.discard(start)
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in neighbors4(r, c):
                if (nr, nc) in remaining:
                    remaining.discard((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def sheet_connected(N, removed):
    total = N * N
    if len(removed) >= total:
        return False
    # find a start cell not removed
    start = None
    for r in range(N):
        for c in range(N):
            if (r, c) not in removed:
                start = (r, c)
                break
        if start is not None:
            break
    if start is None:
        return False
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in removed and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == total - len(removed)


def diversity_target(V):
    return max(2, min(8, V // 6))


def score_from_marks(N, t, V, marks):
    """marks: list of distinct (r,c) cell coordinates in [0,N).
    Returns (feasible, reason, F, area_ratio, hole_count, D_target)."""
    G = group_elements(N, t)
    if len(marks) > V:
        return False, "too many marks", 0.0, 0.0, 0, 0
    if len(set(marks)) != len(marks):
        return False, "duplicate marks", 0.0, 0.0, 0, 0
    for (r, c) in marks:
        if not (0 <= r < N and 0 <= c < N):
            return False, "mark out of range", 0.0, 0.0, 0, 0

    removed = set()
    for (r, c) in marks:
        removed |= orbit(r, c, G)

    if not sheet_connected(N, removed):
        return False, "unfolded sheet disconnected", 0.0, 0.0, 0, 0

    area_ratio = len(removed) / float(N * N)
    target = 1.0 / 3.0
    area_score = max(0.0, 1.0 - abs(area_ratio - target) / target)

    comps = connected_components(removed)
    interior = [comp for comp in comps
                if not any(r == 0 or r == N - 1 or c == 0 or c == N - 1 for (r, c) in comp)]
    shapes = set(canonical_shape(comp) for comp in interior)
    hole_count = len(shapes)
    D_target = diversity_target(V)
    diversity_score = min(1.0, hole_count / D_target)

    F = 0.5 * area_score + 0.5 * diversity_score
    return True, "ok", F, area_ratio, hole_count, D_target


# ---------------------------------------------------------------------------
# Checker's own internal trivial baseline construction (must mirror
# solutions/trivial.py so that the trivial reference lands at ratio ~0.1).
# ---------------------------------------------------------------------------
def baseline_marks(N, t, V):
    r = max(1, N // 4)
    c0 = max(1, (3 * N) // 5)
    avail = max(0, (N - 2) - c0)
    K0 = max(1, min(V // 10, 1 + avail // 2))
    G = group_elements(N, t)
    # deterministic shrink-to-feasible: this row of marks is a "trivial" choice,
    # not a verified-safe one, so fall back to fewer marks if it happens to
    # collide with a fold axis badly enough to disconnect the sheet.
    for k in range(K0, 0, -1):
        cand = [(r, c0 + 2 * i) for i in range(k)]
        removed = set()
        for (rr, cc) in cand:
            removed |= orbit(rr, cc, G)
        if sheet_connected(N, removed):
            return cand
    return [(r, c0)]


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    N, t, V = int(toks[0]), int(toks[1]), int(toks[2])
    return N, t, V


def parse_output(path, V):
    """Strict, bounded parsing.  Rejects malformed / huge / non-finite input."""
    try:
        with open(path) as f:
            data = f.read()
    except Exception:
        return None, "cannot read output"
    toks = data.split()
    if not toks:
        return None, "empty output"
    if len(toks) > 2 * V + 5:
        return None, "output too long"
    try:
        K = int(toks[0])
    except Exception:
        return None, "bad K"
    if K < 0 or K > V:
        return None, "K out of range"
    if len(toks) < 1 + 2 * K:
        return None, "too few tokens for K"
    marks = []
    for i in range(K):
        rt, ct = toks[1 + 2 * i], toks[2 + 2 * i]
        try:
            r = int(rt)
            c = int(ct)
        except Exception:
            return None, "non-integer coordinate"
        if not (math.isfinite(r) and math.isfinite(c)):
            return None, "non-finite coordinate"
        marks.append((r, c))
    return marks, "ok"


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, t, V = read_instance(in_path)

    marks, why = parse_output(out_path, V)
    if marks is None:
        print("Ratio: 0.0  # %s" % why)
        return 0

    feasible, reason, F, area_ratio, hole_count, D_target = score_from_marks(N, t, V, marks)
    if not feasible:
        print("Ratio: 0.0  # %s" % reason)
        return 0

    bmarks = baseline_marks(N, t, V)
    bfeas, breason, B, _, _, _ = score_from_marks(N, t, V, bmarks)
    if not bfeas or B <= 1e-9:
        B = 1e-9

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f area_ratio=%.4f holes=%d D_target=%d Ratio: %.6f" %
          (F, B, area_ratio, hole_count, D_target, ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
