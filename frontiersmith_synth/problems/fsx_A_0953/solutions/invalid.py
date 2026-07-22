# TIER: invalid
# Malformed candidate: returns a predictions list of the wrong length (and never probes).
# The evaluator's strict shape check must reject this -> 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
if inst.get("phase") == "query":
    print(json.dumps({"probes": []}))
else:
    # wrong length on purpose (should equal len(test_queries))
    print(json.dumps({"predictions": [0.0, 0.0, 0.0]}))
