# TIER: trivial
# Do-nothing baseline: zero flux, i.e. "no traffic ever moves". This exactly
# reproduces the checker's own internal baseline construction -> Ratio ~0.1.
import sys

sys.stdin.read()
print("0")
