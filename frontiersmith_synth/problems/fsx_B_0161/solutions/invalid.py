# TIER: invalid
# Emit a grid of the WRONG shape (one row too short) and slip in a non-finite entry.
# Either flaw fails the evaluator's strict validation, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
phases = [[0.0 for _ in range(n)] for _ in range(n - 1)]  # only n-1 rows
if phases:
    phases[0][0] = float("nan")
print(json.dumps({"phases": phases}))
