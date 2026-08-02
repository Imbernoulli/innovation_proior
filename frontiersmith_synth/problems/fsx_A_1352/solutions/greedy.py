# TIER: greedy
import sys

# The obvious first approach: scan triangles in row-major order until a
# panchromatic one turns up (no use of the boundary/parity structure at all).
# Because the output format demands a validated door-to-door certificate for
# ANY credit, this solution still has to build one -- but it HONESTLY reports
# every vertex its scan touched as EXTRA on top of that mandatory certificate,
# so on instances where the panchromatic triangle is far into the scan order
# it pays a large, unnecessary query bill relative to `strong`.


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

    # --- the "obvious" approach: scan for a panchromatic triangle ---
    scanned = set()
    target = None
    for i in range(N):
        for j in range(N - i):
            for t in (up_tri(i, j, N), down_tri(i, j, N)):
                if t is None:
                    continue
                scanned |= set(t)
                cols = {COL[v] for v in t}
                if cols == {0, 1, 2}:
                    target = t
                    break
            if target is not None:
                break
        if target is not None:
            break

    # --- output format still demands the real certificate; build it too ---
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

    extra = scanned - set().union(*[set(t) for t in path])

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
