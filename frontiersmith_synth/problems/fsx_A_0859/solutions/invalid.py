# TIER: invalid
# Emits an expression that references an unknown variable ('y', the reading
# itself, is not something you're allowed to look up when predicting the
# variance FUNCTION of the load) and an unsupported function -> the checker's
# strict grammar validator rejects it and prints Ratio: 0.0.
import sys

sys.stdin.read()
print("wobble(y) + sin(x)")
