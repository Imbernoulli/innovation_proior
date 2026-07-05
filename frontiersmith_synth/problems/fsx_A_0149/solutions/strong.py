# TIER: strong
# Golden-ratio rank-1 Kronecker lattice: x = frac(k * phi), y = (k + 0.5)/N.
# The golden ratio is the "most irrational" multiplier, giving the most even
# 1D fill in x while the stratified y keeps rows balanced -> lowest star
# discrepancy of the ladder. A tiny x offset centres the stream.
import sys
import math

t = sys.stdin.read().split()
N = int(t[0])

phi = (math.sqrt(5.0) - 1.0) / 2.0  # 0.6180339887...

print(N)
out = []
for k in range(N):
    x = (k * phi) % 1.0
    y = (k + 0.5) / N
    out.append("%.12f %.12f" % (x, y))
sys.stdout.write("\n".join(out) + "\n")
