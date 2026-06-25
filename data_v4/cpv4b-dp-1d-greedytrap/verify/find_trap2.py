import random
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
    if n == 0: return 0
    if n == 1:
        return c[0]
    i = 0 if c[0] <= c[1] else 1
    total = c[i]
    while True:
        if i >= n-2:
            break
        a, b = c[i+1], c[i+2]
        i = i+1 if a <= b else i+2
        total += c[i]
    return total

rng = random.Random(7)
found = []
for _ in range(500000):
    n = rng.randint(4, 6)
    c = [rng.randint(0, 9) for _ in range(n)]   # NON-NEGATIVE costs
    g, o = greedy(n,c), optimal(n,c)
    if g > o:
        found.append((n, c, g, o))
        if len(found) >= 6:
            break
for n, c, g, o in found:
    print(f"n={n} c={c} greedy={g} optimal={o}")
