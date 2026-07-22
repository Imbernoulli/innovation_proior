# TIER: invalid
# Emit a malformed assignment: one entry is 2 (not a valid 0/1) and the length is off by
# one. The evaluator rejects any non-{0,1} / mis-shaped answer, so this scores 0.0 on
# every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
assign = [0] * (n - 1) + [2]
print(json.dumps({"assign": assign}))
