# TIER: trivial
# Do-nothing baseline: ignore D, d, and rho entirely and predict a single
# constant discharge rate -- the geometric mean of the training Q column.
# This reproduces the checker's own baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if len(data) < 2:
    print("1.0"); sys.exit(0)
n = int(data[1])
vals = data[2:]
qs = [float(vals[4 * i + 3]) for i in range(n)]
qs = [q for q in qs if q > 0.0]
if not qs:
    print("1.0"); sys.exit(0)
log_mean = sum(math.log(q) for q in qs) / len(qs)
qbar = math.exp(log_mean)
print("%.10g" % qbar)
