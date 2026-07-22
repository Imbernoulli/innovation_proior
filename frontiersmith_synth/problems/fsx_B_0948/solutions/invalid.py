# TIER: invalid
# Emits an expression referencing an unknown function/name -> the checker's
# strict grammar validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("wobble(r) + banana * t")
