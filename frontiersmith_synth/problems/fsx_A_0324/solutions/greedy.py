# TIER: greedy
"""Best-of-many-random: sample many independent random layouts (uniform in the
unit triangle) and keep the one whose faintest triangle is largest. Beats the
single fixed reference layout, but does no local refinement."""
import sys
import random


def min_triangle_area(pts):
    n = len(pts)
    best = float("inf")
    for i in range(n):
        xi, yi = pts[i]
        for j in range(i + 1, n):
            dxj = pts[j][0] - xi
            dyj = pts[j][1] - yi
            for k in range(j + 1, n):
                area = 0.5 * abs(dxj * (pts[k][1] - yi) - (pts[k][0] - xi) * dyj)
                if area < best:
                    best = area
    return best


def rand_layout(rng, n):
    pts = []
    for _ in range(n):
        a = rng.random()
        b = rng.random()
        if a + b > 1.0:
            a, b = 1.0 - a, 1.0 - b
        pts.append((a, b))
    return pts


def main():
    n = int(sys.stdin.read().split()[0])
    rng = random.Random(20240607)
    K = 600
    best_pts = None
    best_val = -1.0
    for _ in range(K):
        pts = rand_layout(rng, n)
        v = min_triangle_area(pts)
        if v > best_val:
            best_val = v
            best_pts = pts
    out = ["%.17g %.17g" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
