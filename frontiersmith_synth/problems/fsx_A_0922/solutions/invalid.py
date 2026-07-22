# TIER: invalid
# Deliberately broken: reads the instance, then emits a routes array with the
# wrong row length (and an out-of-range link index for good measure) so
# every instance must be rejected by the evaluator's shape/range validation.
import sys, json

inst = json.load(sys.stdin)
rounds = inst["rounds"]
K = inst["k"]

routes = [[K + 5] * (rnd["n"] + 1) for rnd in rounds]  # too long AND out-of-range

print(json.dumps({"routes": routes}))
