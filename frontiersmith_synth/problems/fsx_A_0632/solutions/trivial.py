# TIER: trivial
# Do-nothing baseline: ignore load and temperature entirely and predict a
# single constant loss -- the mean of the training y column. This reproduces
# the checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
ys = [float(vals[3 * i + 2]) for i in range(n)]
mn = sum(ys) / len(ys)
print("%.10g" % mn)
