# TIER: trivial
# Do-nothing baseline: ignore interference entirely, predict realized ==
# commanded. This is exactly the checker's own internal baseline construction
# -> reproduces Ratio ~= 0.1 by construction.
import sys

sys.stdin.read()
print("KERNEL 0")
print("OUT v")
