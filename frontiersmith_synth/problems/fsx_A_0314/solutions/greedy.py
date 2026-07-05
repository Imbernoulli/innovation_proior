# TIER: greedy
# Incremental spacing: insert turbines one at a time. Each new turbine is placed at
# the candidate-grid location that maximizes the current thinnest wake triangle.
# Myopic (never revisits earlier placements) -> beats the ring but is not annealed.
import sys, math

LO, HI = 0.02, 0.98

def tri_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

def min_pair_area(pts, cand):
    # min triangle area over triples (i, j, cand)
    m = len(pts)
    best = None
    for i in range(m):
        ai = pts[i]
        for j in range(i + 1, m):
            a = tri_area(ai, pts[j], cand)
            if best is None or a < best:
                best = a
    return best if best is not None else float("inf")

def dist2_min(pts, cand):
    best = None
    for p in pts:
        d = (p[0] - cand[0]) ** 2 + (p[1] - cand[1]) ** 2
        if best is None or d < best:
            best = d
    return best if best is not None else float("inf")

def main():
    data = sys.stdin.read().split()
    n = int(data[0])

    G = 18
    grid = []
    for a in range(G):
        for b in range(G):
            x = LO + (HI - LO) * a / (G - 1)
            y = LO + (HI - LO) * b / (G - 1)
            grid.append((x, y))

    pts = []
    for step in range(n):
        best_c = None
        best_val = -1.0
        for c in grid:
            if len(pts) < 2:
                val = dist2_min(pts, c)          # spread the first two turbines
            else:
                val = min_pair_area(pts, c)       # maximize thinnest new triangle
            if val > best_val:
                best_val = val
                best_c = c
        pts.append(best_c)

    out = ["%.17g %.17g" % (p[0], p[1]) for p in pts]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
