# TIER: trivial
# Do-nothing baseline: guess that the lantern row never changes (the
# "identity" rule cM). This reproduces the checker's own internal baseline
# construction almost exactly -> Ratio ~= 0.1.
import sys

sys.stdin.read()
print("cM == 1")
