# TIER: trivial
# Natural channel order 0,1,...,N-1 -- no reordering at all (reproduces the evaluator's
# weak baseline). Scores ~0.1 by construction.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"order": list(range(N))}))
