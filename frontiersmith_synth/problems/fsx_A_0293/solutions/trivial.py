# TIER: trivial
# Bottom-row baseline: N equal disks in a single row along the valley floor,
# radius r_b = min(W/2N, (H/2 - Q)/2). Clears the wetland by construction and
# reproduces the checker's internal baseline B -> Ratio ~ 0.1.
import sys

t = sys.stdin.read().split()
N = int(t[0]); W = float(t[1]); H = float(t[2])
CX = float(t[3]); CY = float(t[4]); Q = float(t[5])

r_b = min(W / (2.0 * N), (H / 2.0 - Q) / 2.0)
r = r_b * (1.0 - 1e-7)  # tiny shrink to stay strictly feasible under FP

print(N)
out = []
for i in range(N):
    x = (i + 0.5) * W / N
    y = r_b  # sit on the floor at height equal to the (unshrunk) baseline radius
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
