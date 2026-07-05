# TIER: trivial
# Constant predictor: emit the mean of the observed train rate constants.
# Reproduces the checker's internal constant-mean baseline B -> scores ~0.1.
import sys
d = sys.stdin.read().split()
M = int(d[0])
ks = [float(d[3 * i + 3]) for i in range(M)]
mean = sum(ks) / len(ks)
print("%.10e" % mean)
