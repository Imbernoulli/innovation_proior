# TIER: trivial
# Do-nothing baseline: ignore n, m, S2, B2 entirely and predict a single
# constant deviation -- the geometric mean of the training D column. This
# reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("1.0"); sys.exit(0)
rows = int(data[0])
vals = data[2:]
Ds = [float(vals[5 * i + 4]) for i in range(rows)]
gm = math.exp(sum(math.log(d) for d in Ds) / len(Ds))
print("%.10g" % gm)
