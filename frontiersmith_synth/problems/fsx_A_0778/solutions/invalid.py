# TIER: invalid
# Flat match at the maximum allowed rate applied to EVERY donor from dollar zero.
# Well-formed as a schedule (K=1, rate in [0,R_MAX]) but pays out far more than the
# stated budget on any instance -> the checker's feasibility gate must reject it (score 0).
import sys

t = sys.stdin.read().split()
N = int(t[0])
K_MAX = int(t[1])
R_MAX = float(t[2])
print("1")
print("%.6f" % R_MAX)
