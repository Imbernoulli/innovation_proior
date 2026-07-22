# TIER: invalid
# Malformed candidate: returns a prediction of the wrong length (and never probes).
# The evaluator's strict shape check must reject this -> 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
if inst.get("phase") == "query":
    print(json.dumps({"queries": []}))
else:
    # wrong length on purpose (should be G)
    print(json.dumps({"pred": [0.0, 0.0, 0.0]}))
