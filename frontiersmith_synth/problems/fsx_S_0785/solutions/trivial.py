# TIER: trivial
# Do-nothing-clever baseline: order-1 local delta persistence.  Predict the
# next platform by repeating the most recently observed step (h1 + (h1-h2)).
# This is EXACTLY the checker's own internal baseline strategy -> reproduces
# it closely, landing at ratio ~0.1. Ignores the training log entirely.
import sys

sys.stdin.read()
print("SLOT1 h1 + ( h1 - h2 )")
print("SLOT2 NONE")
