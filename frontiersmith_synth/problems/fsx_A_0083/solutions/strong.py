# TIER: strong
# Hexagonal (offset-row) packing: find the largest equal radius r such that at least
# N disks fit on a hex lattice inside [r, S-r]^2, then deploy the first N. Denser than
# an aligned square grid, so the total radius is larger for the same drone budget.
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])


def hex_points(r):
    if r <= 0:
        return []
    dy = math.sqrt(3.0) * r
    pts = []
    row = 0
    y = r
    while y <= S - r + 1e-12:
        offset = r if (row % 2 == 1) else 0.0
        x = r + offset
        while x <= S - r + 1e-12:
            pts.append((x, y))
            x += 2.0 * r
        y += dy
        row += 1
    return pts


# binary search: largest r with count(r) >= N
lo, hi = 1e-9, S / 2.0
for _ in range(80):
    mid = 0.5 * (lo + hi)
    if len(hex_points(mid)) >= N:
        lo = mid
    else:
        hi = mid

r = lo
pts = hex_points(r)
# fall back to a safe square grid if something degenerate happened
if len(pts) < N:
    k = int(math.ceil(math.sqrt(N)))
    r = 0.5 * S / k
    pts = []
    for i in range(N):
        pts.append(((i % k + 0.5) * S / k, (i // k + 0.5) * S / k))

r_out = r * (1.0 - 1e-6)  # shrink radius; positions unchanged -> strictly non-overlapping
print(N)
out = []
for i in range(N):
    x, y = pts[i]
    out.append("%.10f %.10f %.10f" % (x, y, r_out))
sys.stdout.write("\n".join(out) + "\n")
