# TIER: invalid
# Emits a policy for only 3 of the 6 trace_ids (and repeats trace_id 0), which
# fails validation (trace_ids must be exactly the set {0..5}) -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
period = inst["period"]

policies = [
    {"trace_id": 0, "level": [10.0] * period, "trend": 0.0, "react": 0.0},
    {"trace_id": 0, "level": [10.0] * period, "trend": 0.0, "react": 0.0},
    {"trace_id": 1, "level": [10.0] * period, "trend": 0.0, "react": 0.0},
]

print(json.dumps({"policies": policies}))
