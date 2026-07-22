# TIER: invalid
# Emits an infeasible artifact: a "walk" that steps onto a cell that is not
# even orthogonally adjacent to the previous one (teleporting across the
# map) and then revisits its own start cell -- both are hard feasibility
# violations, so the checker must reject with Ratio 0.0.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); V = int(it[p + 1]); E = int(it[p + 2]); S = int(it[p + 3]); p += 4
    cells = []
    for i in range(V):
        r = int(it[p]); c = int(it[p + 1]); rew = int(it[p + 2])
        k = int(it[p + 3]); kid = int(it[p + 4]); p += 5
        cells.append((r, c, rew, k, kid))
    sr, sc = cells[S][0], cells[S][1]
    # pick some far-away declared cell to "teleport" to (non-adjacent jump),
    # then revisit the start cell (self-avoiding violation).
    far = max(cells, key=lambda t: abs(t[0] - sr) + abs(t[1] - sc))
    out = ["3", "%d %d" % (sr, sc), "%d %d" % (far[0], far[1]), "%d %d" % (sr, sc)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
