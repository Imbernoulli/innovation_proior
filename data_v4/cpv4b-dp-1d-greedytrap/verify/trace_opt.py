from functools import lru_cache
import sys

def optimal_path(n, c):
    INF = float("inf")
    @lru_cache(maxsize=None)
    def from_stone(i):
        res = (0, []) if i >= n-2 else (INF, None)
        for step in (1,2):
            nxt = i+step
            if 0 <= nxt < n:
                v, path = from_stone(nxt)
                if c[nxt] + v < res[0]:
                    res = (c[nxt] + v, [nxt] + path)
        return res
    cand = []
    v0, p0 = from_stone(0)
    cand.append((c[0] + v0, [0] + p0))
    if n >= 2:
        v1, p1 = from_stone(1)
        cand.append((c[1] + v1, [1] + p1))
    best = min(cand, key=lambda t: t[0])
    return best

for line in sys.stdin:
    parts = line.split()
    if not parts: continue
    n = int(parts[0]); c = list(map(int, parts[1:1+n]))
    cost, path = optimal_path(n, c)
    print(f"n={n} c={c} cost={cost} landings(0-indexed)={path}")
