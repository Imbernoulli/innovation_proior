# TIER: invalid
# Emit a structurally invalid hash modulus (M=0). The evaluator's validity
# check requires 2 <= M <= MAX_M, so every instance is rejected outright and
# scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"a": 1, "c": 0, "M": 0}))
