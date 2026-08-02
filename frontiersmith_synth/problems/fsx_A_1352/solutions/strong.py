# TIER: strong
import sys

# Insight: the panchromatic triangle can be found by a single door-to-door
# walk (Sperner's parity-argument constructive proof) starting at the unique
# boundary "0-1 door" on the z=0 edge, instead of scanning the whole grid.
# We report ONLY the walk itself (no EXTRA lookups), so USED == OPT exactly.


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); D = int(toks[1])
    COL = {}
    p = 2
    for _ in range((N + 1) * (N + 2) // 2):
        x = int(toks[p]); y = int(toks[p + 1]); c = int(toks[p + 2])
        COL[(x, y)] = c
        p += 3

    # find the unique boundary door on z=0 (exploits the boundary condition:
    # colors along this edge are monotone 0...0 1...1, so the flip is unique).
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

    out = []
    ax, ay = cur[0]; bx, by = cur[1]; cx, cy = cur[2]
    out.append(f"ANSWER {ax} {ay} {bx} {by} {cx} {cy}")
    out.append(f"PATH {len(path)}")
    for tri in path:
        vals = []
        for (x, y) in tri:
            vals += [x, y]
        out.append(" ".join(map(str, vals)))
    out.append("EXTRA 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
