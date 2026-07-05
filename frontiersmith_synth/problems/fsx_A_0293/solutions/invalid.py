# TIER: invalid
# Emits N large overlapping disks stacked at the valley centre: they mutually
# overlap AND flood the wetland -> infeasible -> Ratio: 0.0.
import sys

t = sys.stdin.read().split()
N = int(t[0]); W = float(t[1]); H = float(t[2])
CX = float(t[3]); CY = float(t[4]); Q = float(t[5])

print(N)
out = []
for i in range(N):
    out.append("%.6f %.6f %.6f" % (CX, CY, 0.45 * H))
sys.stdout.write("\n".join(out) + "\n")
