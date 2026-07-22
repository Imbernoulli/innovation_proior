# TIER: trivial
# Do-nothing baseline: ignore T and h entirely and predict a single constant
# magnetization -- the geometric mean of the training m column.  This
# reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
ms = [float(vals[3 * i + 2]) for i in range(n)]
gm = math.exp(sum(math.log(x) for x in ms) / len(ms))
print("%.10g" % gm)
