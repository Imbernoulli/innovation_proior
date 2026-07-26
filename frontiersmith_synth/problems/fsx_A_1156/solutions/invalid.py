# TIER: invalid
# Emits an expression referencing a disallowed function and an unknown name ->
# the strict AST whitelist in the checker rejects it and prints Ratio: 0.0.
import sys
sys.stdin.read()
print("exp(t) + wobble * gearRatio")
