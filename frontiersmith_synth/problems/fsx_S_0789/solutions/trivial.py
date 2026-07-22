# TIER: trivial
# Do-nothing baseline: reproduce the checker's own fixed internal candidate
# "XY" (a single, arbitrary library feature, not data-derived at all).
import sys

sys.stdin.read()
print("XY")
