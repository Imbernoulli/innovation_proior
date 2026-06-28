#!/usr/bin/env python3
# Random SMALL-case generator. Usage: python3 gen.py <seed>
# Keeps n small (<= 16) so the exhaustive brute-force oracle stays fast, and S small
# so the DP array is tiny. Mixes feasible and infeasible instances, plus S=0 and
# zero/large-weight items to probe the corner cases.
import sys
import random

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rnd = random.Random(seed)

mode = rnd.randint(0, 6)

if mode == 0:
    # tiny, likely feasible: build S as a sum of a few of the weights
    n = rnd.randint(1, 8)
    w = [rnd.randint(1, 12) for _ in range(n)]
    cnt = rnd.randint(0, n)
    picked = rnd.sample(range(n), cnt) if cnt > 0 else []
    S = sum(w[i] for i in picked)
elif mode == 1:
    # S = 0 corner (empty subset is the answer)
    n = rnd.randint(0, 8)
    w = [rnd.randint(0, 12) for _ in range(n)]
    S = 0
elif mode == 2:
    # include zero-weight and oversize items
    n = rnd.randint(1, 10)
    w = [rnd.choice([0, 0, rnd.randint(1, 15), rnd.randint(1, 30)]) for _ in range(n)]
    S = rnd.randint(0, 30)
elif mode == 3:
    # all equal weights (multiplicity matters for min count)
    n = rnd.randint(1, 10)
    v = rnd.randint(1, 7)
    w = [v] * n
    S = rnd.randint(0, n * v + 3)
elif mode == 4:
    # n = 0 (no items)
    n = 0
    w = []
    S = rnd.randint(0, 5)
else:
    # general small random
    n = rnd.randint(1, 12)
    w = [rnd.randint(0, 20) for _ in range(n)]
    S = rnd.randint(0, 40)

# Build output
print(n, S)
print(" ".join(map(str, w)))
