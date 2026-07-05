# TIER: invalid
# Emits a single huge disk that spills past the strip boundary (containment violation)
# -> the checker scores it 0.
import sys

t = sys.stdin.read().split()
N = int(t[0]); L = float(t[1]); W = float(t[2])
print(1)
# radius = W centred at mid-height: y + r = 1.5W > W -> infeasible
print("%.6f %.6f %.6f" % (L / 2.0, W / 2.0, W))
