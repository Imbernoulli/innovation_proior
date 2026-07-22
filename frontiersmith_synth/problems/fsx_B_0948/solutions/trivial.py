# TIER: trivial
# Do-nothing baseline: predict the checker's own constant-1.2 baseline,
# ignoring r and t entirely -> reproduces ~0.1 by construction.
import sys

sys.stdin.read()
print("1.2")
