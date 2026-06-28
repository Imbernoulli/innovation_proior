#!/usr/bin/env python3
# Brute oracle for "MST with exactly k white edges".
# Obviously-correct method: enumerate all subsets of (n-1) edges, keep those that
# form a spanning tree (connect all n vertices) with exactly k white edges, take
# the minimum total raw weight. Exponential in m -> only for small instances.
import sys
from itertools import combinations

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it)); m = int(next(it)); k = int(next(it))
    except StopIteration:
        return
    edges = []
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        w = int(next(it))
        c = int(next(it))
        edges.append((u, v, w, c))

    if k < 0 or k > n - 1:
        print(-1); return

    def connects(subset):
        # union-find over the chosen edges; must connect all n vertices with n-1 edges
        par = list(range(n))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]; x = par[x]
            return x
        cnt = n
        for (u, v, w, c) in subset:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False  # cycle => not a tree
            par[ru] = rv; cnt -= 1
        return cnt == 1

    if n == 1:
        # single vertex: the only spanning tree has 0 edges
        print(0 if k == 0 else -1); return

    best = None
    need = n - 1
    for subset in combinations(edges, need):
        white = sum(c for (_, _, _, c) in subset)
        if white != k:
            continue
        if connects(subset):
            tot = sum(w for (_, _, w, _) in subset)
            if best is None or tot < best:
                best = tot
    print(best if best is not None else -1)

if __name__ == "__main__":
    main()
