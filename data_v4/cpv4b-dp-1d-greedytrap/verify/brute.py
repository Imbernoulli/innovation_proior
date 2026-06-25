import sys
from functools import lru_cache

def solve(data):
    it = iter(data)
    n = int(next(it))
    c = [int(next(it)) for _ in range(n)]

    if n == 0:
        return 0

    INF = float("inf")

    # Legal walk model (independent re-derivation by brute enumeration):
    #   - You start on the near bank. Your FIRST landing must be on stone 0 or
    #     stone 1 (you cannot leap clear over every stone from the bank).
    #   - From stone i your next landing is stone i+1 or stone i+2.
    #   - You may step off to the far bank only from stone n-1 or stone n-2
    #     (a +1 or +2 leap that reaches index >= n), at no extra cost.
    # Cost of a walk = sum of c[] over the stones landed on. Minimize it.
    #
    # We enumerate by reachable stone index. start is modeled separately so that
    # the first move is restricted to {stone 0, stone 1}.

    @lru_cache(maxsize=None)
    def from_stone(i):
        # min additional cost to get from standing on stone i to the far bank
        if i >= n - 2:
            # from stone n-1 or n-2 a single leap clears to the far bank
            res = 0
        else:
            res = INF
        for step in (1, 2):
            nxt = i + step
            if 0 <= nxt < n:
                res = min(res, c[nxt] + from_stone(nxt))
        return res

    # first landing: stone 0 or stone 1 (if they exist)
    best = c[0] + from_stone(0)
    if n >= 2:
        best = min(best, c[1] + from_stone(1))
    return best

def main():
    data = sys.stdin.read().split()
    print(solve(data))

if __name__ == "__main__":
    main()
