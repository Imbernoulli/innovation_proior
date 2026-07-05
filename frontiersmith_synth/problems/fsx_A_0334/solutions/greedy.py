# TIER: greedy
"""Best-of-many random uniform layouts: sample many independent random station
sets and keep the one whose combined d_min/d_max ratio is largest. Seeded by the
instance size so it is deterministic. Beats the diagonal baseline, but its
per-test behaviour differs from the structured 'strong' solver."""
import sys
import math
import random


def uratio(pts):
    m = len(pts)
    dmin = float("inf")
    dmax = 0.0
    for i in range(m):
        xi, yi = pts[i]
        for j in range(i + 1, m):
            dx = xi - pts[j][0]
            dy = yi - pts[j][1]
            d = dx * dx + dy * dy
            if d < dmin:
                dmin = d
            if d > dmax:
                dmax = d
    if dmax <= 0:
        return 0.0
    return math.sqrt(dmin) / math.sqrt(dmax)


def main():
    tok = sys.stdin.read().split()
    idx = 0
    n = int(tok[idx]); idx += 1
    k = int(tok[idx]); idx += 1
    land = []
    for _ in range(k):
        land.append((float(tok[idx]), float(tok[idx + 1]))); idx += 2

    rng = random.Random(9000 + n)
    best = None
    best_u = -1.0
    for _ in range(400):
        st = [(rng.uniform(0.03, 0.97), rng.uniform(0.03, 0.97)) for _ in range(n)]
        u = uratio(land + st)
        if u > best_u:
            best_u = u
            best = st

    out = ["%.6f %.6f" % (x, y) for x, y in best]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
