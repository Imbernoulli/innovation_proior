# TIER: trivial
# Do nothing: never shed.  This is exactly the evaluator's DO-NOTHING reference, so it
# scores 0.1 on every instance -- the whole cascade is absorbed or lost with no operator
# intervention at all.
import sys, json

inst = json.load(sys.stdin)
L, T = inst["L"], inst["T"]
print(json.dumps({"shed": [[0.0] * L for _ in range(T)]}))
