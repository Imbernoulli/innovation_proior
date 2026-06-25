import random, subprocess, sys
from functools import lru_cache

def optimal(n, c):
    if n == 0: return 0
    INF = float("inf")
    @lru_cache(maxsize=None)
    def from_stone(i):
        res = 0 if i >= n-2 else INF
        for step in (1,2):
            nxt = i+step
            if 0 <= nxt < n:
                res = min(res, c[nxt] + from_stone(nxt))
        return res
    best = c[0] + from_stone(0)
    if n >= 2:
        best = min(best, c[1] + from_stone(1))
    from_stone.cache_clear()
    return best

def greedy(n, c):
    # "obvious" greedy: from the start, pick the cheaper of stone 0 / stone 1;
    # then from each stone hop to the cheaper of the next two stones; cost = sum landed.
    if n == 0: return 0
    # choose first stone
    if n == 1:
        i = 0; total = c[0]
    else:
        if c[0] <= c[1]:
            i = 0
        else:
            i = 1
        total = c[i]
    while i < n-2 if False else True:
        if i >= n-2:
            break  # can step to far bank
        a, b = c[i+1], c[i+2]
        if a <= b:
            i = i+1
        else:
            i = i+2
        total += c[i]
    return total

rng = random.Random(12345)
found = []
for _ in range(200000):
    n = rng.randint(3, 7)
    c = [rng.randint(-6, 9) for _ in range(n)]
    if greedy(n, c) != optimal(n, c) and greedy(n,c) > optimal(n,c):
        found.append((n, c, greedy(n,c), optimal(n,c)))
        if len(found) >= 8:
            break

for n, c, g, o in found:
    print(f"n={n} c={c} greedy={g} optimal={o}")
