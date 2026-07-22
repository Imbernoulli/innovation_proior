# TIER: invalid
import sys, json

inst = json.load(sys.stdin)
# out-of-range kp (cap is 320) -> the evaluator's strict validator must reject this -> score 0
print(json.dumps({"kp": 1.0e9, "kd": 0.5, "ki": 0.0, "resonators": []}))
