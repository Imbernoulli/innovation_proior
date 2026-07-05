# TIER: trivial
# Single center row of N equal disks along the highway centerline. Reproduces the
# checker baseline B exactly (radius r0 = min(W/4, L/(2N))), so it scores ~0.1.
import sys

t = sys.stdin.read().split()
N = int(t[0]); L = float(t[1]); W = float(t[2])
r0 = min(W / 4.0, L / (2.0 * N)) * (1.0 - 1e-7)
print(N)
out = []
for i in range(N):
    x = (i + 0.5) * L / N
    y = W / 2.0
    out.append("%.10f %.10f %.10f" % (x, y, r0))
sys.stdout.write("\n".join(out) + "\n")
