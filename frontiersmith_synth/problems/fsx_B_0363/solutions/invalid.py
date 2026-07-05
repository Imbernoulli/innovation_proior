# TIER: invalid
# Emits a phase array of the WRONG shape (a single flat row instead of N x N).
# The evaluator's strict shape check rejects it -> score 0. (A real "cheat"
# that just claims a high metric also fails: the evaluator re-simulates.)
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"phase": [0.0] * N}))
