import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx+=1
    m = int(data[idx]); idx+=1
    crews = []
    for _ in range(m):
        mk = int(data[idx]); idx+=1
        c = int(data[idx]); idx+=1
        crews.append((mk, c))
    FULL = (1<<n) - 1

    if n == 0:
        print(0)
        return

    INF = float('inf')

    # Independent brute force: a recursive exact-cover search.
    # We process the LOWEST uncovered module and try every crew whose mask:
    #   (a) is a subset of the still-uncovered set, and
    #   (b) covers that lowest uncovered module.
    # We branch on the choice of crew, mark its modules covered, recurse.
    # Each crew may be used at most once; we pass an index set / used flags.
    # Because masks chosen along a path are pairwise disjoint (each is a subset of
    # the remaining-uncovered set at the time it is chosen), every crew can in fact
    # appear at most once on a path even without explicit used-tracking; but we still
    # forbid re-use to match the contract exactly by indexing crews.
    valid = [(mk, c) for (mk, c) in crews if 1 <= mk <= FULL]

    @lru_cache(maxsize=None)
    def solve(remaining):
        if remaining == 0:
            return 0
        low = remaining & (-remaining)
        lowbit_index = low  # the lowest uncovered module bit
        bestv = INF
        for mk, c in valid:
            if (mk & remaining) == mk and (mk & lowbit_index):
                sub = solve(remaining ^ mk)
                if sub < INF:
                    v = c + sub
                    if v < bestv:
                        bestv = v
        return bestv

    ans = solve(FULL)
    print(-1 if ans == INF else ans)

main()
