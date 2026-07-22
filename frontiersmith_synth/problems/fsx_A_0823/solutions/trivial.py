# TIER: trivial
# Do-nothing baseline: ignore w entirely and predict a single constant
# amplitude -- the arithmetic mean of the training |X1| values.  This
# reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
amps = []
for i in range(n):
    xre = float(vals[3 * i + 1])
    xim = float(vals[3 * i + 2])
    amps.append(math.sqrt(xre * xre + xim * xim))
c = sum(amps) / len(amps)
print("%.10g" % c)
