# TIER: trivial
# Predict every well log VALID (constant). Ignores all features -> baseline calibration ~0.1.
import sys, json
inst = json.load(sys.stdin)
m = inst["m"]
print(json.dumps({"w": [0.0] * m, "b": 1e9}))
