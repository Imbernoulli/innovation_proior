# TIER: greedy
# Put every layout ON the DTLZ2 sphere (distance vars = 0.5, so g = 0) and
# scatter the angle vars with a fixed deterministic pseudo-random sequence.
# This already beats the lone centre point, but random angles clump and leave
# gaps on the front -> sub-optimal hypervolume coverage.
import sys, json, random
inst = json.load(sys.stdin)
M, k, n, budget = inst["M"], inst["k"], inst["n"], inst["budget"]
rng = random.Random(1234 + M * 100 + budget)
pts = []
for _ in range(budget):
    x = [0.0] * n
    for j in range(M - 1):        # angle vars: random in [0,1]
        x[j] = rng.random()
    for j in range(M - 1, n):     # distance vars: pinned on the sphere
        x[j] = 0.5
    pts.append(x)
print(json.dumps({"points": pts}))
