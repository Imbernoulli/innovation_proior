# TIER: trivial
# Ignore the instance entirely: inoculate every cell with species index 0. This
# reproduces the evaluator's own fixed reference layout exactly, so it always
# normalizes to ~0.1.
import sys, json

inst = json.load(sys.stdin)
L = inst["L"]
assign = [0] * L
print(json.dumps({"assign": assign}))
