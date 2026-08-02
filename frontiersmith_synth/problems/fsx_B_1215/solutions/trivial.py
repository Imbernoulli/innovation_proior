# TIER: trivial
# Do-nothing baseline: ignore ACC/INFL/SEIS entirely and predict a single
# constant probability -- the training catalogue's own eruption rate. This
# reproduces the checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.5"); sys.exit(0)
n = int(data[0])
vals = data[2:]
ys = [int(vals[4 * i + 3]) for i in range(n)]
r = sum(ys) / len(ys) if ys else 0.5
print("%.10g" % r)
