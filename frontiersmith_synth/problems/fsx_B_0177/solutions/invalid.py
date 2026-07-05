# TIER: invalid
# Emits a flat 1-D phase list instead of the required N-by-N grid -> rejected by the
# shape gate -> scores 0 on every instance.
import sys, json
inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"phase": [0.0] * N}))
