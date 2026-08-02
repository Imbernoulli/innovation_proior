# TIER: trivial
# Do-nothing baseline: ignore every input and predict a single constant calving
# time -- the geometric mean of the training T column. This reproduces the
# checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
Ts = [float(vals[6 * i + 5]) for i in range(n)]
gm = math.exp(sum(math.log(v) for v in Ts) / len(Ts))
print("%.10g" % gm)
