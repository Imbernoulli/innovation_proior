# TIER: trivial
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); k = int(next(it))
    forced = {}
    for _ in range(k):
        r = int(next(it)); c = int(next(it)); v = int(next(it))
        forced[(r, c)] = v
    A = [[(1 if j >= i else -1) for j in range(n)] for i in range(n)]
    for (r, c), v in forced.items():
        A[r][c] = v
    print("\n".join(" ".join(map(str, row)) for row in A))

main()
