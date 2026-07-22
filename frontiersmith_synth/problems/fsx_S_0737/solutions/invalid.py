# TIER: invalid
# Infeasible artifact: uses arithmetic addition (not in the allowed grammar)
# and, even if it were parsed, evaluates outside {0,1} for some neighbourhood
# patterns (e.g. cL=cM=cR=1 -> 3). Must score 0.
import sys

sys.stdin.read()
print("cL + cM + cR")
