# TIER: invalid
# Emits an expression using the raw ** operator (disallowed -- the grammar
# requires powv) and an unknown variable -> the strict AST whitelist in the
# checker rejects it and prints Ratio: 0.0.
import sys
sys.stdin.read()
print("D**2 + banana * rho")
