# TIER: trivial
# Reproduces the checker's own internal baseline: give every pair exactly
# one relay, parked at the straight-line midpoint between its source and
# destination. Ignores interference and hop-count trade-offs entirely.
import sys


def main():
    t = sys.stdin.read().split()
    p = 0
    m = int(t[p]); p += 1
    R = int(t[p]); p += 1
    p += 5  # P, alpha, N0, Xmax, Ymax (unused)
    pairs = []
    for _ in range(m):
        sx = float(t[p]); sy = float(t[p + 1]); dx = float(t[p + 2]); dy = float(t[p + 3])
        p += 4
        pairs.append((sx, sy, dx, dy))

    out = []
    for (sx, sy, dx, dy) in pairs:
        mx, my = (sx + dx) / 2.0, (sy + dy) / 2.0
        out.append("1 %.6f %.6f" % (mx, my))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
