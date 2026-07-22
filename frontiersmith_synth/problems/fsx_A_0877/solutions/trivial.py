# TIER: trivial
# Transmit nothing. Reproduces the evaluator's own weak "all-zero input"
# reference exactly, so this scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"x": [0.0] * n}))
