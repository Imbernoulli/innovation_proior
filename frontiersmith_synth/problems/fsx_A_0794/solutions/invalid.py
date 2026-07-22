# TIER: invalid
# Emits an expression referencing an unknown function and an unknown
# variable -> the strict AST whitelist in the checker rejects it and prints
# Ratio: 0.0.
import sys
sys.stdin.read()
print("wobble(B2) + banana * n")
