# TIER: invalid
# Emits non-finite step sizes -> the evaluator rejects the schedule -> score 0.
import sys, json
inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"eta": [float("nan")] * K, "gamma": [float("inf")] * K}))
