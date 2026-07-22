# TIER: invalid
# Emits "feature*feature", which the checker's grammar explicitly forbids
# (only coefficient*feature is allowed -- the library is a set of names to
# take a LINEAR combination of, not to multiply together) -> Ratio: 0.0.
import sys

sys.stdin.read()
print("XX*YY")
