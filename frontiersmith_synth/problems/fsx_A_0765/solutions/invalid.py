# TIER: invalid
# Emits an expression that references a name outside the allowed variable
# set (n, g, D, M, A, K, H) -- the checker's AST whitelist must reject it,
# scoring 0.
import sys

sys.stdin.read()
print("2.0*beauty_index + D")
