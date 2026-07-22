# TIER: invalid
# Emits an expression referencing an unknown function and an unknown name ->
# the strict AST whitelist in the checker rejects it and prints Ratio: 0.0.
import sys
sys.stdin.read()
print("wobble(x1) + banana * x3")
