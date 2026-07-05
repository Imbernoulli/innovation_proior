# TIER: trivial
# Constant predictor: emit the mean of the observed train power draws.
# Reproduces the checker's internal baseline B -> scores ~0.1.
import sys
d = sys.stdin.read().split()
M = int(d[0])
ps = [float(d[3 * i + 3]) for i in range(M)]
mean = sum(ps) / len(ps)
print("%.6f" % mean)
