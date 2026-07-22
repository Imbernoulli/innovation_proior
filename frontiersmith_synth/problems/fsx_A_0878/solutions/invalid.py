# TIER: invalid
# Malformed policy: window < 2 (violates the required [2,50] range) and
# init_trust out of [0,1] and decay given as a non-numeric string. The
# evaluator's strict validator rejects this shape -> that instance scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({
    "mode": "blend",
    "init_trust": 5.0,
    "decay": "fast",
    "window": 1,
    "edge_threshold": 0.5,
    "reset_on_edge": "yes",
}))
