# TIER: greedy
# Crude "magnitude" foraging curve f(s)=|s|: a real nonlinearity, but it throws away the sign of
# the stimulus, so it underperforms a well-shaped curve.
import sys, json
inst = json.load(sys.stdin)
grid = inst["grid"]
print(json.dumps({"ys": [abs(float(x)) for x in grid]}))
