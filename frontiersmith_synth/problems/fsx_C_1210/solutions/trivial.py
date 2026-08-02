# TIER: trivial
# Do-nothing baseline: ignore cohort identity, age, and calendar day entirely and
# predict a single constant lift -- the mean of every visible row. Reproduces the
# checker's own constant baseline -> Ratio ~ 0.1.
import sys

data = sys.stdin.read().split()
if not data:
    print("0.05"); sys.exit(0)
n = int(data[0])
vals = data[2:]
lifts = [float(vals[5 * i + 4]) for i in range(n)]
mean_l = sum(lifts) / len(lifts)
print("%.10g" % mean_l)
