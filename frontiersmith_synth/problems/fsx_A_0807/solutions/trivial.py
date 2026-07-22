# TIER: trivial
# Do-nothing baseline: ignore all four knobs and predict a single constant
# response -- the geometric mean of the training y column. This reproduces
# the checker's own constant baseline -> Ratio ~ 0.1.
import sys, math

data = sys.stdin.read().split()
if not data:
    print("0.0"); sys.exit(0)
n = int(data[0])
# tokens: n, testid, then 3 rows x 4 ints (grading matrix) = 12 tokens, then n*5 data tokens
vals = data[2 + 12:]
ys = [float(vals[5 * i + 4]) for i in range(n)]
gm = math.exp(sum(math.log(v) for v in ys) / len(ys))
print("%.10g" % gm)
