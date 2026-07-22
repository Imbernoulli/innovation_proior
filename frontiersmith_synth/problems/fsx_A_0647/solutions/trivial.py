# TIER: trivial
"""One Wang-tile TYPE per grid CELL ("the map"): a boustrophedon (snake) path
visits every cell of the k x (T+1) target rectangle exactly once; each step
binds via one brand-new, single-use glue (strength 2) to the previous cell.
This is always feasible and reproduces the checker's own internal baseline
construction exactly -> ratio ~= 0.1 by definition. It never reuses a tile
type, so its cost is the full cell count (area), the classic O(N^2)-style
"just draw the picture" recipe.
"""
import sys


def main():
    T = int(sys.stdin.read().split()[0])
    k = max(1, T.bit_length())
    height = (T + 1) if k == 1 else (2 * T + 1)

    # boustrophedon path over the k x height grid, starting at (0,0); height
    # is always odd for k > 1 so the last row is traversed ascending and ends
    # at column k-1, letting the extra "flag" row above columns 1..k-1 (see
    # verify.py) attach as one final westward leg back to column 1.
    path = []
    for y in range(height):
        xs = range(k) if y % 2 == 0 else range(k - 1, -1, -1)
        for x in xs:
            path.append((x, y))
    if k > 1:
        assert path[-1] == (k - 1, height - 1)
        for x in range(k - 1, 0, -1):
            path.append((x, height))

    n = len(path)
    types = []  # each: [Nl,Ns,El,Es,Sl,Ss,Wl,Ws]
    label = 1

    def empty_row():
        return [0, 0, 0, 0, 0, 0, 0, 0]

    rows = [empty_row() for _ in range(n)]
    IDX = {"N": 0, "E": 2, "S": 4, "W": 6}

    def dir_between(a, b):
        ax, ay = a
        bx, by = b
        if bx == ax + 1 and by == ay:
            return "E"
        if bx == ax - 1 and by == ay:
            return "W"
        if by == ay + 1 and bx == ax:
            return "N"
        if by == ay - 1 and bx == ax:
            return "S"
        raise AssertionError("non-adjacent path step")

    OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
    for i in range(n - 1):
        a, b = path[i], path[i + 1]
        d_ab = dir_between(a, b)
        d_ba = OPP[d_ab]
        lab = label
        label += 1
        j = IDX[d_ab]
        rows[i][j] = lab
        rows[i][j + 1] = 2
        j2 = IDX[d_ba]
        rows[i + 1][j2] = lab
        rows[i + 1][j2 + 1] = 2

    out = [str(n)]
    for r in rows:
        out.append(" ".join(map(str, r)))
    out.append("1")  # seed = first cell in the path = (0,0)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
