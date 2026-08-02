import sys
import random

# ---------------------------------------------------------------------------
# Sperner coloring generator ("Finding the small colorful triangle").
#
# Grid: barycentric triangle of side N -- vertices (x,y,z), x+y+z=N, x,y,z>=0.
# Coloring: label(x,y,z) = argmax_k (D*v_k - p_k), lowest index wins ties,
# for a fixed interior point P=(p0,p1,p2)/D chosen per test.  This is the
# classical "nearest-corner-direction-from-P" Sperner coloring: it AUTOMATICALLY
# satisfies the Sperner boundary rule (face v_k=0 never gets label k) whenever
# every p_k > 0, and the unique small triangle containing P is panchromatic.
#
# We bias P (large z-share) so the interior color-transition -- and hence the
# forced door-to-door walk from the boundary -- has to travel deep into the
# grid (OPT scales with N), while a row-major scan for a panchromatic triangle
# can be made to touch a much larger fraction of the O(N^2) triangles first.
# ---------------------------------------------------------------------------

D = 8
LADDER = [15, 20, 28, 38, 50, 65, 85, 110, 140, 180]


def label(x, y, z, p):
    vals = (D * x - p[0], D * y - p[1], D * z - p[2])
    best = 0
    if vals[1] > vals[best]:
        best = 1
    if vals[2] > vals[best]:
        best = 2
    return best


def up_tri(i, j, N):
    if i + j > N - 1:
        return None
    return ((i, j), (i + 1, j), (i, j + 1))


def down_tri(i, j, N):
    if i + j > N - 2:
        return None
    return ((i + 1, j), (i, j + 1), (i + 1, j + 1))


def col_of(x, y, N, p):
    return label(x, y, N - x - y, p)


def count_panchromatic(N, p):
    cnt = 0
    for i in range(N):
        for j in range(N - i):
            t = up_tri(i, j, N)
            cols = {col_of(x, y, N, p) for (x, y) in t}
            if cols == {0, 1, 2}:
                cnt += 1
    for i in range(N - 1):
        for j in range(N - 1 - i):
            t = down_tri(i, j, N)
            cols = {col_of(x, y, N, p) for (x, y) in t}
            if cols == {0, 1, 2}:
                cnt += 1
    return cnt


def find_start_i(N, p):
    prev = None
    for x in range(N, -1, -1):
        y = N - x
        c = col_of(x, y, N, p)
        if prev is not None and prev != c and {prev, c} == {0, 1}:
            return x
        prev = c
    return None


def walk_len(N, p):
    """Distinct vertices used by the forced door-to-door walk (checker's OPT)."""
    i0 = find_start_i(N, p)
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
        cols = {col_of(x, y, N, p) for (x, y) in cur}
        if cols == {0, 1, 2}:
            return len(visited)
        edges = [frozenset([cur[a], cur[b]]) for a, b in [(0, 1), (1, 2), (0, 2)]]
        doors = []
        for e in edges:
            pts = list(e)
            c0 = col_of(pts[0][0], pts[0][1], N, p)
            c1 = col_of(pts[1][0], pts[1][1], N, p)
            if {c0, c1} == {0, 1}:
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


def search_P(N, seed, tries=60):
    rng = random.Random(seed)
    best = None
    for _ in range(tries):
        frac2 = rng.uniform(0.35, 0.75)
        p2 = max(D, int(frac2 * D * N))
        rem = D * N - p2
        if rem < 2 * D:
            continue
        p0 = rng.randint(D, rem - D)
        p1 = rem - p0
        if p1 < D:
            continue
        p = (p0, p1, p2)
        if count_panchromatic(N, p) != 1:
            continue
        opt = walk_len(N, p)
        if opt is None:
            continue
        if best is None or opt > best[1]:
            best = (p, opt)
    return best


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N = LADDER[(idx - 1) % len(LADDER)]
    seed = 2000 + idx
    found = search_P(N, seed, tries=60)
    if found is None:
        # deterministic, extremely conservative fallback (should not trigger
        # in practice -- kept only as a safety net for robustness).
        found = search_P(N, seed + 1, tries=200)
    p, opt = found

    out = [f"{N} {D}"]
    for x in range(N, -1, -1):
        for y in range(N - x + 1):
            c = col_of(x, y, N, p)
            out.append(f"{x} {y} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
