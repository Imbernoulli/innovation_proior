# TIER: trivial
# Do-nothing baseline: ignore the histogram, the cache parameters, and the
# working-set size entirely -- predict a single constant miss rate, the mean
# of the training rows.  Since every training row sits deep in the sub-cliff
# regime, this constant is tiny (near 0).  It reproduces the checker's own
# constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
t, C, A, M, n_train = (int(x) for x in data[:5])
idx = 5 + M
mrs = []
for i in range(n_train):
    mrs.append(float(data[idx + 2 * i + 1]))
mean_mr = sum(mrs) / len(mrs) if mrs else 0.0
print("%.10g" % mean_mr)
