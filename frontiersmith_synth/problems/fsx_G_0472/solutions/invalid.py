# TIER: invalid
# Emit an out-of-range momentum coefficient (alpha must satisfy -1 < alpha < 1).
# alpha = 1.0 fails the strict range check, so the evaluator rejects the answer
# and scores every instance 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"eta_x": 0.05, "eta_y": 0.05, "theta": 0.0, "alpha": 1.0}))
