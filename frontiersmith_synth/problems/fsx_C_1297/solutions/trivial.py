# TIER: trivial
# Submit a well-formed but empty controller: no rules at all, so every
# (state, sensed-token) pair falls back to the evaluator's default WAIT.
# The robot never moves, never reaches the relay beacon, and scores the
# floor value on every one of the 10 grids.
import sys, json

inst = json.load(sys.stdin)

answer = {"start_state": 0, "rules": []}
print(json.dumps(answer))
