# TIER: invalid
# Route EVERY platoon through gantry 0.  Uses only one gantry (great objective!) but
# overflows its per-cycle capacity C, so the routing is infeasible -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
platoons = inst["platoons"]
print(json.dumps({"assign": [0 for _ in platoons]}))
