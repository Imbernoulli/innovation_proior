# TIER: trivial
# Identity activation phi(x) = x: just echo the grid back as the table.  This is
# exactly the evaluator's linear-collapse reference, so every battery scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
grid = inst["grid"]
print(json.dumps({"y": grid}))
