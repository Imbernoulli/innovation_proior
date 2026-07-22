# TIER: invalid
# Emit a transition that points at state n (one past the last valid index 0..n-1).
# The evaluator rejects any out-of-range transition target -> every device scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = 2
delta = [[0, 1], [1, n]]  # n is out of range for an n-state automaton
print(json.dumps({"delta": delta, "start": 0, "accept": [1]}))
