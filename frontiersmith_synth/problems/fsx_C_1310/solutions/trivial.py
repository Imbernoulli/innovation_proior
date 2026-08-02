# TIER: trivial
# Image only the first THREE targets in the order they appear in the input (no
# sorting, no cost/decay/cloud reasoning at all), attempted in pass 1 only, pass 2
# left empty. A small, dumb, fixed-size plan.
import sys, json

inst = json.load(sys.stdin)
ids = [t["id"] for t in inst["targets"][:3]]
print(json.dumps({"pass1": ids, "pass2": []}))
