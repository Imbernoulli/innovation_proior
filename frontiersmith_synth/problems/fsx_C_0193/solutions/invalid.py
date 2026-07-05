# TIER: invalid
# Emits a partition of the WRONG length -> the evaluator's shape validation
# rejects it and every city scores exactly 0.0.
import sys, json

inst = json.load(sys.stdin)
n = int(inst["n"])
print(json.dumps([0] * (n // 2)))
