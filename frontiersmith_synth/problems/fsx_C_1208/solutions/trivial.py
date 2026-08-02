# TIER: trivial
# Do-nothing baseline: ignore the period and the exposure log entirely --
# predict a single constant engagement rate, the mean of the training rows.
# This reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
tid, n_train = int(data[0]), int(data[1])
idx = 2
es = []
for i in range(n_train):
    es.append(float(data[idx + 3 * i + 2]))
mean_e = sum(es) / len(es) if es else 0.0
print("%.10g" % mean_e)
