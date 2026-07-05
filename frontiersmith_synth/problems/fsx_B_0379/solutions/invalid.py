# TIER: invalid
# Ill-formed mask: emit a grid with the wrong number of columns (N-1 instead of
# N). The grader validates the phase shape strictly, so this is rejected and
# scores 0 -- demonstrating the feasibility gate.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
phase = [[0.0] * (N - 1) for _ in range(N)]
print(json.dumps({"phase": phase}))
