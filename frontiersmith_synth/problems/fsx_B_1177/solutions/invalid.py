# TIER: invalid
# Emits an expression that references an unknown variable name -> the
# checker's strict AST validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("wobble * f + banana")
