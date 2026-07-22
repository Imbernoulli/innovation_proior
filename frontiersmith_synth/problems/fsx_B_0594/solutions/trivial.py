# TIER: trivial
# Do nothing: leave every wall as-is. Reproduces the checker's do-nothing baseline (~0.1).
import sys
toks = sys.stdin.read().split()
M = int(toks[1])
print(" ".join(["0"] * M))
