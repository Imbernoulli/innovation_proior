# TIER: trivial
import sys, json

inst = json.load(sys.stdin)
# plant-naive, fixed, unit-ish gains -- ignores m/c/k and the entire published disturbance suite
print(json.dumps({"kp": 1.5, "kd": 0.5, "ki": 0.0, "resonators": []}))
