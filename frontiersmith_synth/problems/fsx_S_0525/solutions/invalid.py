# TIER: invalid
# Emits an infeasible design: a material index out of range and a thickness beyond dmax.
# The checker must reject it -> Ratio 0.0.
import sys
it = sys.stdin.read().split()
# M is token index 2
M = int(it[2])
print(2)
print("%d %.6f" % (M + 5, 999999.0))   # bad material index + oversize thickness
print("%d %.6f" % (0, 1e18))
