# TIER: strong
# Hexagonal (offset-row) packing over the valley floor, dropping any lattice disk
# that would flood the wetland. Hex rows pack denser than a square grid, so for the
# same reservoir budget N a larger common radius fits -> larger total dam frontage.
# Binary-search the largest r with at least N obstacle-clear hex disks, keep the N
# farthest from the wetland.
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); W = float(t[1]); H = float(t[2])
CX = float(t[3]); CY = float(t[4]); Q = float(t[5])


def hex_points(r):
    if r <= 0:
        return []
    dy = math.sqrt(3.0) * r
    pts = []
    row = 0
    y = r
    while y <= H - r + 1e-12:
        offset = r if (row % 2 == 1) else 0.0
        x = r + offset
        while x <= W - r + 1e-12:
            dx = x - CX; dyy = y - CY
            if math.sqrt(dx * dx + dyy * dyy) >= r + Q - 1e-9:
                pts.append((x, y))
            x += 2.0 * r
        y += dy
        row += 1
    return pts


lo, hi = 1e-9, min(W, H) / 2.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if len(hex_points(mid)) >= N:
        lo = mid
    else:
        hi = mid

r = lo
pts = hex_points(r)
if len(pts) < N:
    r = min(W / (2.0 * N), (H / 2.0 - Q) / 2.0)
    pts = [((i + 0.5) * W / N, r) for i in range(N)]

pts.sort(key=lambda p: -((p[0] - CX) ** 2 + (p[1] - CY) ** 2))
pts = pts[:N]

r_out = r * (1.0 - 1e-6)
print(len(pts))
out = ["%.10f %.10f %.10f" % (x, y, r_out) for (x, y) in pts]
sys.stdout.write("\n".join(out) + "\n")
