# TIER: invalid
# Emits an expression calling an unknown function and referencing an unknown
# variable -> the strict AST whitelist in the checker rejects it and prints
# Ratio: 0.0.
import sys

sys.stdin.read()
print("wobble(k) + drift * k**2")
