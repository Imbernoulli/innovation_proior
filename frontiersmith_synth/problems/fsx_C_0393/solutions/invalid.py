# TIER: invalid
# Emit a schedule of the WRONG length (and with a non-finite gain), so the evaluator's strict
# validation rejects it -> 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)

# length 3 regardless of T, and a NaN gain -> fails shape + finiteness checks
print(json.dumps({"a": [float("nan"), 1.0, 2.0], "b": [0.0, 0.0, 0.0]}))
