import sys
from functools import lru_cache

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]
    idx += n
    if n == 0:
        print(0)
        return

    # Fully independent brute force: actually simulate every legal sequence of
    # adjacent merges on the *current row*, trying all positions each step,
    # and take the minimum total cost.  State = tuple of current stone values.
    # We memoize on the row tuple so deep cases still finish, but the search
    # itself makes no use of the interval-DP decomposition.

    seen = {}

    def solve(row):
        if len(row) == 1:
            return 0
        key = row
        if key in seen:
            return seen[key]
        best = None
        for i in range(len(row) - 1):
            left = row[i]
            right = row[i + 1]
            cost = (left | right)
            merged = left ^ right
            nxt = row[:i] + (merged,) + row[i + 2:]
            tot = cost + solve(nxt)
            if best is None or tot < best:
                best = tot
        seen[key] = best
        return best

    print(solve(tuple(a)))

main()
