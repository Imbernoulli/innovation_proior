# TIER: invalid
# Emits an expression that references a forbidden name -> the checker's
# strict AST validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("floor + amplitude * n ** (-alpha)")
