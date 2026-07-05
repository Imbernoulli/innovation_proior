# TIER: greedy
# Aligned square-grid packing of equal disks over the whole valley floor, dropping
# any grid cell whose disk would flood the wetland. Binary-search the largest disk
# radius r such that at least N grid disks still fit (and clear the obstacle), then
# deploy the N of them farthest from the wetland (keeps the strongest cells).
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); W = float(t[1]); H = float(t[2])
CX = float(t[3]); CY = float(t[4]); Q = float(t[5])


def grid_points(r):
    if r <= 0:
        return []
    pts = []
    y = r
    while y <= H - r + 1e-12:
        x = r
        while x <= W - r + 1e-12:
            dx = x - CX; dy = y - CY
            if math.sqrt(dx * dx + dy * dy) >= r + Q - 1e-9:
                pts.append((x, y))
            x += 2.0 * r
        y += 2.0 * r
    return pts


lo, hi = 1e-9, min(W, H) / 2.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if len(grid_points(mid)) >= N:
        lo = mid
    else:
        hi = mid

r = lo
pts = grid_points(r)
if len(pts) < N:
    # degenerate fallback: bottom row baseline
    r = min(W / (2.0 * N), (H / 2.0 - Q) / 2.0)
    pts = [((i + 0.5) * W / N, r) for i in range(N)]

# keep the N cells farthest from the wetland centre
pts.sort(key=lambda p: -((p[0] - CX) ** 2 + (p[1] - CY) ** 2))
pts = pts[:N]

r_out = r * (1.0 - 1e-6)
print(len(pts))
out = ["%.10f %.10f %.10f" % (x, y, r_out) for (x, y) in pts]
sys.stdout.write("\n".join(out) + "\n")
