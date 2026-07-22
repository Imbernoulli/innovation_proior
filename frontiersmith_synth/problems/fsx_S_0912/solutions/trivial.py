# TIER: trivial
# Do-nothing genome: a single NOP byte. Casts no votes at all, so every cell
# defaults to type 0. Matches the evaluator's own normalization anchor exactly
# (by construction), so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"tape": [0]}))
