# TIER: trivial
# Do-nothing baseline: quote the full appraisal, i.e. assume no haggling ever
# happens. This is exactly the checker's own internal baseline construction,
# so it reproduces Ratio ~= 0.1 by construction.
import sys

sys.stdin.read()
print("v")
