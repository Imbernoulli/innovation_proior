# TIER: invalid
# Emits an over-budget, out-of-range allocation (every wall stuffed with 999 layers).
# The checker's feasibility gate must reject this -> score 0.
import sys
toks = sys.stdin.read().split()
M = int(toks[1])
print(" ".join(["999"] * M))
