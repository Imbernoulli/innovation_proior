# TIER: invalid
# Emits a malformed ranking: for every query it returns the SAME fixed list
# [0, 1, ..., N-2] regardless of the query.  For any query i > 0 this list is not a
# permutation of {0..N-1}\{i} (it contains i itself and omits N-1), so the evaluator
# rejects the submission and scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
row = list(range(N - 1))
print(json.dumps({"ranking": [list(row) for _ in range(N)]}))
