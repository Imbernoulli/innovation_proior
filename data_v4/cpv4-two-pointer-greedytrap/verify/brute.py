import sys
from functools import lru_cache

def solve(data):
    it = iter(data.split())
    n = int(next(it))
    L = int(next(it))
    w = [int(next(it)) for _ in range(n)]

    # Independent maximum-matching brute force.
    # A cart is an unordered pair {i,j} with w[i]+w[j] <= L. We want the
    # maximum number of vertex-disjoint such pairs: a maximum matching on the
    # "compatibility graph". For small n we compute it by exhaustive recursion:
    # take the first unused person, either leave them unpaired or pair them
    # with some compatible later person, and recurse.
    used = [False] * n

    def rec():
        # find the first unused person; pair them or leave them out, recurse.
        i = 0
        while i < n and used[i]:
            i += 1
        if i >= n:
            return 0
        used[i] = True
        best = rec()  # leave person i unpaired (i is now "used up" / removed)
        for j in range(i + 1, n):
            if not used[j] and w[i] + w[j] <= L:
                used[j] = True
                best = max(best, 1 + rec())
                used[j] = False
        used[i] = False
        return best

    return rec()

if __name__ == "__main__":
    print(solve(sys.stdin.read()))
