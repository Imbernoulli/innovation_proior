# TIER: invalid
# Emits an expression that references an unknown name -> the checker's strict
# whitelist AST validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("haggle(n, v) + v")
