# TIER: trivial
# Predict the training-label MEAN for every point (zero weights, bias = mean).
# This reproduces the evaluator's own baseline construction exactly, so it scores
# ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
d = inst["d"]
y = inst["y_train"]
b = sum(y) / len(y)
print(json.dumps({"w": [0.0] * d, "b": b}))
