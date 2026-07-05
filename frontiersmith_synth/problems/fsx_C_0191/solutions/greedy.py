# TIER: greedy
# Greedy curve: a plain tanh with unit gain. It introduces real curvature and beats the
# linear baseline on several halls, but its soft unit-gain squashing under-uses the knot
# range and saturates early, so it under-performs a better-shaped (steeper-gain) curve.
import sys, json, math

inst = json.load(sys.stdin)
grid = inst["grid"]
ys = [math.tanh(x) for x in grid]
print(json.dumps({"activation": ys}))
