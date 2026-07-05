# TIER: invalid
# Emits a non-finite prediction (inf) for a masked cell. The evaluator rejects any
# non-finite value, so every instance scores 0.
import sys, json

inst = json.load(sys.stdin)
mask = inst["masked"]
preds = [float("inf")] * len(mask)
print(json.dumps({"preds": preds}))
