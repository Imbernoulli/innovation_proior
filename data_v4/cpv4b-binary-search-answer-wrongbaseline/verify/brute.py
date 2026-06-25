import sys
from itertools import combinations

def solve(data):
    it = iter(data)
    n = next(it); k = next(it); L = next(it)
    p = [next(it) for _ in range(n)]
    # Choose exactly k of the n posts. Clearance = min over cyclic-adjacent
    # chosen pairs of the arc distance between them (going around the ring).
    # Maximize the clearance.
    best = -1
    for comb in combinations(range(n), k):
        pts = sorted(p[i] for i in comb)
        # cyclic gaps
        m = len(pts)
        mn = None
        for i in range(m):
            a = pts[i]
            b = pts[(i + 1) % m]
            gap = (b - a) % L
            if mn is None or gap < mn:
                mn = gap
        if mn > best:
            best = mn
    return best

def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))

if __name__ == "__main__":
    main()
