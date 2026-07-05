# TIER: trivial
# Top-block pile-up: drop the K emitters on the K plots whose surrounding block
# has the largest total deficit, ignoring overlap.  On a clustered vineyard these
# all crowd onto the single hottest zone and their wetted squares overlap heavily,
# so the union recovered is small.  This reproduces the evaluator's weak reference
# exactly, scoring ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N, R, K = inst["N"], inst["R"], inst["K"]
grid = inst["grid"]

# 2D prefix sums
P = [[0] * (N + 1) for _ in range(N + 1)]
for r in range(N):
    acc = 0
    for c in range(N):
        acc += grid[r][c]
        P[r + 1][c + 1] = P[r][c + 1] + acc


def block(r, c):
    r0 = max(0, r - R); c0 = max(0, c - R)
    r1 = min(N, r + R + 1); c1 = min(N, c + R + 1)
    return P[r1][c1] - P[r0][c1] - P[r1][c0] + P[r0][c0]


# deterministic tie-break identical to the evaluator: (-win, r*N + c)
order = sorted(((-block(r, c), r * N + c, r, c) for r in range(N) for c in range(N)))
emitters = [[t[2], t[3]] for t in order[:K]]

print(json.dumps({"emitters": emitters}))
