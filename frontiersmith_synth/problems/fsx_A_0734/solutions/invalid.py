# TIER: invalid
# Emits a schedule with a discount depth ABOVE max_discount for every segment/week
# (and the wrong number of segments' worth of rows is not even attempted -- just a
# blatantly out-of-range depth). Fails validation on every instance -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n_weeks = inst["n_weeks"]
dmax = inst["max_discount"]
m = len(inst["segments"])

bad_depth = dmax + 0.5
schedule = [[bad_depth] * n_weeks for _ in range(m)]
print(json.dumps({"schedule": schedule}))
