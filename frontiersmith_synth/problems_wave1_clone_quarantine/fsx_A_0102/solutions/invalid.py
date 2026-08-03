# TIER: invalid
# Cram every act onto a single stage.  Every instance in this family has a total
# footprint that far exceeds one stage's capacity C (and more than K acts), so
# stage 0 is both over capacity AND over the changeover-window cap -> the layout is
# infeasible -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"assign": [0] * N}))
