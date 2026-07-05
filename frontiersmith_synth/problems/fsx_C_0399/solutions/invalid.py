# TIER: invalid
# Emit a schedule of the WRONG length (T+3 entries) with a non-list "m" field.
# The evaluator's strict shape check rejects it -> every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"a": [0.1] * (T + 3), "m": "fast", "o": [0.0] * T}))
