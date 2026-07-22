# TIER: trivial
"""Per-cell tiling: one unique, unshared tile type for every grid cell.
Row 0 chains west->east on unique horizontal glues; every other row chains
south->north on unique vertical glues. Always correct, always O(n*W) types --
exactly the checker's own baseline construction."""
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    W = max(1, (n - 1).bit_length()) if n >= 2 else 1

    def idx(r, c):
        return r * n + c

    lines = []
    for r in range(W):
        for c in range(n):
            north = ("S_%d_%d" % (r + 1, c), 2) if r + 1 < W else (".", 0)
            south = ("S_%d_%d" % (r, c), 2) if r > 0 else (".", 0)
            east = ("L_0_%d" % (c + 1), 2) if (r == 0 and c + 1 < n) else (".", 0)
            west = ("L_0_%d" % c, 2) if (r == 0 and c > 0) else (".", 0)
            value = (c >> r) & 1
            lines.append((idx(r, c), north, south, east, west, value))

    out = [str(len(lines))]
    for tid, N, S, E, Wg, val in lines:
        out.append("%d %s %d %s %d %s %d %s %d %d" %
                    (tid, N[0], N[1], S[0], S[1], E[0], E[1], Wg[0], Wg[1], val))
    out.append(str(idx(0, 0)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
