# TIER: invalid
# Emits an expression that calls a function and references an unknown name --
# the checker's strict AST whitelist (arithmetic + p1,p2,d only, no calls, no
# other names) rejects it -> Ratio: 0.0.
import sys

sys.stdin.read()
print("sqrt(p1) + banana * d")
