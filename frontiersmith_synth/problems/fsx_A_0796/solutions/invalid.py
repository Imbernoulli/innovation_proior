# TIER: invalid
# Emits an expression that references a name outside the allowed variable
# set (n, s) -- the checker's AST whitelist must reject it, scoring 0.
import sys

sys.stdin.read()
print("2.0*n/K + 5*s")
