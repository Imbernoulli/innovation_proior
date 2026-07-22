# TIER: trivial
# Do-nothing baseline: ignore n and s entirely and predict a single constant --
# the arithmetic mean of the training register readings. This reproduces the
# checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n_rows = int(data[0])
vals = data[2:]
ys = [float(vals[3 * i + 2]) for i in range(n_rows)]
mean_y = sum(ys) / len(ys)
print("%.10g" % mean_y)
