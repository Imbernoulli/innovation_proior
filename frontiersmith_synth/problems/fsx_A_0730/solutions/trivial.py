# TIER: trivial
# Always guess zero hits: a single-state machine whose transition table maps
# every symbol back to itself, and whose output table always reports 0. This
# exactly reproduces the evaluator's weak reference, so it scores ~0.1 on
# every instance regardless of the stream.
import sys, json

inst = json.load(sys.stdin)
m = inst["m"]

trans = [[0] * m]
out = [0.0]

print(json.dumps({"n_states": 1, "start": 0, "trans": trans, "out": out}))
