# TIER: trivial
# Do-nothing baseline: ignore time and temperature entirely and predict a
# single constant displacement -- the mean of the training d column. This
# reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
vals = data[2:]
ds = [float(vals[3 * i + 2]) for i in range(n)]
mn = sum(ds) / len(ds)
print("%.10g" % mn)
