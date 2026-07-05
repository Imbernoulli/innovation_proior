# TIER: trivial
# Diagonal baseline: N wells evenly along the field diagonal. Reproduces the
# checker's internal baseline, so it scores about 0.1.
import sys

t = sys.stdin.read().split()
N = int(t[0])
print(N)
out = []
for k in range(N):
    v = (k + 0.5) / N
    out.append("%.12f %.12f" % (v, v))
sys.stdout.write("\n".join(out) + "\n")
