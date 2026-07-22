# TIER: trivial
# Do-nothing baseline: ignore every column and predict a single constant --
# the arithmetic mean of the training beauty scores. This reproduces the
# checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
Bs = [float(vals[8 * i + 7]) for i in range(n)]
mean_B = sum(Bs) / len(Bs)
print("%.10g" % mean_B)
