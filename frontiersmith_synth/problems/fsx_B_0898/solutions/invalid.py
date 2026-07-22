# TIER: invalid
import sys, json

inst = json.load(sys.stdin)
# missing required keys / wrong types -> evaluator must reject and score 0.0
print(json.dumps({"base_target": float("nan"), "trigger": "not-a-number"}))
