# TIER: greedy
# Aligned square grid of equal disks, radius = S/(2*k), k = ceil(sqrt(N)).
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])
k = int(math.ceil(math.sqrt(N)))
r = (0.5 * S / k) * (1.0 - 1e-7)
print(N)
out = []
for i in range(N):
    col = i % k
    row = i // k
    x = (col + 0.5) * S / k
    y = (row + 0.5) * S / k
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
