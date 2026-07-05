# TIER: invalid
# Wrong shape: emits a flat list instead of the required per-group predictions,
# and out-of-range values.  The evaluator must reject this and score 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"predictions": [999, 999, 999]}))
