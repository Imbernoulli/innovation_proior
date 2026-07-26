# TIER: invalid
# Emits an expression using a disallowed function call -> the checker's
# strict grammar (arithmetic + numeric constants over S, C ONLY, no calls)
# rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("S * exp(C) + 1.0")
