# TIER: trivial
# All-zero offsets: set every signal's green phase to 0.  The grid field is uniform,
# so the flow spectrum collapses to a single peak at the DC center and every off-center
# target corridor stays dark.  This is exactly the evaluator's weak baseline, so it
# scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
phases = [[0.0 for _ in range(n)] for _ in range(n)]
print(json.dumps({"phases": phases}))
