# TIER: invalid
# Emits a disk that spills outside the sector (containment violation) -> scores 0.
import sys

t = sys.stdin.read().split()
N = int(t[0]); S = float(t[1])
print(1)
# radius 0.9 centred at the middle: x + r = 1.4 > S -> infeasible
print("%.6f %.6f %.6f" % (S / 2.0, S / 2.0, 0.9 * S))
