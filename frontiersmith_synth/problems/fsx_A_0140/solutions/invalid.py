# TIER: invalid
# Drop every emitter off the edge of the vineyard (coordinates == N, which is out
# of the valid range 0..N-1).  The evaluator rejects any out-of-range coordinate,
# so the whole answer is infeasible and the instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
N, K = inst["N"], inst["K"]

print(json.dumps({"emitters": [[N, N] for _ in range(K)]}))
