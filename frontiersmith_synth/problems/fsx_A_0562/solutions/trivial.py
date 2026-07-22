# TIER: trivial
# Do-nothing baseline: ignore the units and the physics entirely and predict a
# single constant drag force -- the geometric mean of the training F column.
# This reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
Fs = [float(vals[5 * i + 4]) for i in range(n)]
gm = math.exp(sum(math.log(f) for f in Fs) / len(Fs))
print("%.10g" % gm)
