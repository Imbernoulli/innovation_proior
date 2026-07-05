# TIER: trivial
# Identity permutation p[i] = i. This is exactly the checker's internal
# baseline (all displacement vectors lie on the diagonal (h,h)), so it scores
# the calibrated ~0.1.
import sys

n = int(sys.stdin.read().split()[0])
print(" ".join(str(i) for i in range(n)))
