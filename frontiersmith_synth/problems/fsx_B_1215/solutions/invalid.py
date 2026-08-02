# TIER: invalid
# Emits an expression referencing an unknown variable and an unknown function
# -> the strict AST whitelist in the checker rejects it and prints Ratio: 0.0.
import sys
sys.stdin.read()
print("wobble(ACC) + magma_flux * U")
