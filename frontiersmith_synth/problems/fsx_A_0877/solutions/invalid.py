# TIER: invalid
# Blast every symbol at twice the amplitude budget. Since xmax > 0 on every
# instance, this violates |x_t| <= xmax for every symbol -> the evaluator
# rejects the whole answer -> the instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
xmax = inst["xmax"]

print(json.dumps({"x": [2.0 * xmax] * n}))
