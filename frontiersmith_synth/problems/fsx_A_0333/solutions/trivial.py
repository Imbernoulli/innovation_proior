# TIER: trivial
# Reproduce the checker's baseline: give EVERY station the same uniform radius,
# centred on its platform. The common radius is min(smallest wall distance,
# half the smallest inter-platform distance) -- always feasible.
import sys, math


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    pts = []
    idx = 1
    for _ in range(N):
        pts.append((float(toks[idx]), float(toks[idx + 1]))); idx += 2

    minb = min(min(x, 1.0 - x, y, 1.0 - y) for (x, y) in pts)
    mind = None
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if mind is None or d < mind:
                mind = d
    if mind is None:
        mind = 2.0 * minb
    runi = max(0.0, min(0.5 * mind, minb))

    out = []
    for i in range(N):
        out.append("%.9f %.9f %.9f" % (pts[i][0], pts[i][1], runi))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
