# TIER: trivial
# Do-nothing baseline: predict the same context-free constant (0.5) regardless
# of the data -- reproduces the checker's own internal baseline construction.
import sys

sys.stdin.read()
print("0.5")
