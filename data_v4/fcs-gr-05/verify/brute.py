#!/usr/bin/env python3
"""Brute-force oracle for 2-edge-connectivity augmentation.

Slow but obviously correct: read the connected graph, then try to add k edges
for k = 0, 1, 2, ... and check (by a direct definition of "bridge") whether the
augmented graph has any bridge. The first k for which some addition removes all
bridges is the answer.

A bridge is an edge whose removal increases the number of connected components.
We test every edge by removing it and comparing connectivity counts directly.
"""
import sys
from itertools import combinations


def components(n, edges):
    """Number of connected components over vertices 1..n with multiset `edges`."""
    par = list(range(n + 1))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for (u, v) in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            par[ru] = rv
    roots = set(find(i) for i in range(1, n + 1))
    return len(roots)


def has_bridge(n, edges):
    """True if some edge is a bridge (removal increases component count)."""
    base = components(n, edges)
    for idx in range(len(edges)):
        rest = edges[:idx] + edges[idx + 1:]
        if components(n, rest) > base:
            return True
    return False


def solve(n, edges):
    # Drop self-loops (never bridges, irrelevant).
    edges = [(u, v) for (u, v) in edges if u != v]

    if not has_bridge(n, edges):
        return 0

    # Candidate new edges: every unordered pair of distinct vertices.
    pairs = list(combinations(range(1, n + 1), 2))

    for k in range(1, n + 1):
        for combo in combinations(pairs, k):
            if not has_bridge(n, edges + list(combo)):
                return k
    return -1  # unreachable for a connected graph


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return
    edges = []
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        edges.append((u, v))
    print(solve(n, edges))


if __name__ == "__main__":
    main()
