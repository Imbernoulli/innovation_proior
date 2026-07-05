# TIER: invalid
# Stow every fragment into a single hauler (index 0).  For any real shift the total
# mass vastly exceeds one hauler's capacity, so hauler 0 is over capacity and the
# plan fails validation -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
n = len(inst["masses"])

print(json.dumps({"assign": [0] * n}))
