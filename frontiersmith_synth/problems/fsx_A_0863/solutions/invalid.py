# TIER: invalid
# Malformed candidate: claims a giant addition that blows straight through
# V_max on the very first round (also never reads the instance at all).
# The evaluator's strict feasibility check (V + add <= V_max) must reject
# this -> 0.0 on every instance.
import sys, json

json.load(sys.stdin)
print(json.dumps({"add": 1.0e9}))
