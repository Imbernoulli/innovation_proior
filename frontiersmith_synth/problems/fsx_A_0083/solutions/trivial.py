# TIER: trivial
# Row baseline: N equal disks in a single line across the sector. F = S/2.
import sys

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])
r = (S / (2.0 * N)) * (1.0 - 1e-7)  # tiny shrink to stay strictly feasible under FP
print(N)
out = []
for i in range(N):
    x = (i + 0.5) * S / N
    y = S / 2.0
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
