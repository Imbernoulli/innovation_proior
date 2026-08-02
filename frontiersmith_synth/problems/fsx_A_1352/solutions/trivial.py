# TIER: trivial
import sys

# The most naive possible construction: query EVERY vertex of the grid (a
# full exhaustive scan, never stopping early) to be "sure" of the answer,
# then still has to produce the mandatory certificate. This reproduces the
# checker's own worst-case reference: query cost is close to the whole grid.


def up_tri(i, j, N):
    if i + j > N - 1:
        return None
    return ((i, j), (i + 1, j), (i, j + 1))


def down_tri(i, j, N):
    if i + j > N - 2:
        return None
    return ((i + 1, j), (i, j + 1), (i + 1, j + 1))


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); D = int(toks[1])
    COL = {}
    p = 2
    for _ in range((N + 1) * (N + 2) // 2):
        x = int(toks[p]); y = int(toks[p + 1]); c = int(toks[p + 2])
        COL[(x, y)] = c
        p += 3

    # --- exhaustively touch every vertex in the grid, no early stop ---
    target = None
    for i in range(N):
        for j in range(N - i):
            for t in (up_tri(i, j, N), down_tri(i, j, N)):
                if t is None:
                    continue
                cols = {COL[v] for v in t}
                if cols == {0, 1, 2} and target is None:
                    target = t
    all_grid = set(COL.keys())

    # --- mandatory certificate (same construction as strong/greedy) ---
    i0 = None
    prev = None
    for x in range(N, -1, -1):
        y = N - x
        c = COL[(x, y)]
        if prev is not None and prev != c and {prev, c} == {0, 1}:
            i0 = x
            break
        prev = c
    j0 = N - i0 - 1
    cur = ((i0, j0), (i0 + 1, j0), (i0, j0 + 1))
    path = [cur]
    entry_edge = frozenset([cur[1], cur[2]])
    while True:
        cols = {COL[v] for v in cur}
        if cols == {0, 1, 2}:
            break
        edges = [frozenset([cur[a], cur[b]]) for a, b in [(0, 1), (1, 2), (0, 2)]]
        doors = [e for e in edges if {COL[list(e)[0]], COL[list(e)[1]]} == {0, 1}]
        exit_edge = [e for e in doors if e != entry_edge][0]
        third = [v for v in cur if v not in exit_edge][0]
        pA, pB = list(exit_edge)
        newv = (pA[0] + pB[0] - third[0], pA[1] + pB[1] - third[1])
        cur = (pA, pB, newv)
        path.append(cur)
        entry_edge = exit_edge

    extra = all_grid - set().union(*[set(t) for t in path])

    out = []
    ax, ay = cur[0]; bx, by = cur[1]; cx, cy = cur[2]
    out.append(f"ANSWER {ax} {ay} {bx} {by} {cx} {cy}")
    out.append(f"PATH {len(path)}")
    for tri in path:
        vals = []
        for (x, y) in tri:
            vals += [x, y]
        out.append(" ".join(map(str, vals)))
    out.append(f"EXTRA {len(extra)}")
    for (x, y) in extra:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
