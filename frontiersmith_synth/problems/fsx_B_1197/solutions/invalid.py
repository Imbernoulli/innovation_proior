# TIER: invalid
# Emits an expression referencing an unknown name/function -> the checker's
# strict grammar validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("wobble ( t ) + banana * tanh ( t )")
