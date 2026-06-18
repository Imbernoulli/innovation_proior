# Hopcroft-Karp Maximum Bipartite Matching

## Problem

Given a bipartite graph `G = (L, R, E)`, find a maximum-cardinality matching: a largest set of edges with no shared endpoint.

## Method

The algorithm keeps a valid matching in `match_l` and `match_r`. It repeatedly searches for augmenting paths, because by Berge's criterion a matching is maximum exactly when no augmenting path exists.

Each phase does two things:

1. `build_layers()` runs a BFS from all free left vertices through alternating edges. It records left-side layers in `level` and the first reachable free-right distance in `free_level`; expansion beyond that distance is skipped, so the phase is restricted to shortest augmenting paths.
2. `find_path(u)` runs a DFS inside that layered graph. It only advances to a matched left vertex in the next layer, and it only accepts a free right vertex exactly at `free_level`. Successful and failed left vertices are removed from the current phase search, so the phase augments along a maximal vertex-disjoint set of shortest augmenting paths.

The phase-increase proof is the key invariant. Let a phase choose `t` vertex-disjoint shortest augmenting paths, each of length `ell`, and let `M'` be the matching after flipping them. If a later augmenting path `P` for `M'` is disjoint from those paths, then it was already an augmenting path for `M`, so maximality rules out `|P| <= ell`. If `P` touches a chosen path, the shared vertex is internal to `P`, and the unique `M'`-matched edge at that vertex lies on both paths. Thus `(chosen paths) xor P` has at most `t ell + |P| - 1` edges, but it is also the symmetric difference between `M` and a matching of size `|M| + t + 1`, so it contains at least `t + 1` old augmenting paths, each of length at least `ell`. Therefore `|P| >= ell + 1`, and bipartiteness makes the next possible length at least `ell + 2`. Splitting the run after `floor(sqrt(|M*|))` phases, every remaining augmenting path contains at least `floor(sqrt(|M*|))` current-matching edges, so the remaining deficit is `O(sqrt(|M*|))`. Thus there are `O(sqrt V)` phases, each scanning `O(E)` edges, for total time `O(E sqrt V)` and space `O(V + E)`.

## Code

```python
import sys


def read_bipartite(data):
    """Parse n (left size), m (right size), e (edge count), then e edges
    (u in [0, n), v in [0, m)) into a 0-based adjacency list adj[u] -> [v, ...]."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    e = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(e):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
    return n, m, adj


def max_matching(n, m, adj):
    """Return maximum matching size and matched (left, right) pairs."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), n + m + 10))
    match_l = [-1] * n
    match_r = [-1] * m
    INF = n + m + 1
    level = [INF] * n
    free_level = INF

    def build_layers():
        nonlocal free_level
        queue = []
        head = 0
        free_level = INF
        for u in range(n):
            if match_l[u] == -1:
                level[u] = 0
                queue.append(u)
            else:
                level[u] = INF
        while head < len(queue):
            u = queue[head]
            head += 1
            next_level = level[u] + 1
            if next_level >= free_level:
                continue
            for v in adj[u]:
                w = match_r[v]
                if w == -1:
                    free_level = next_level
                elif level[w] == INF:
                    level[w] = next_level
                    queue.append(w)
        return free_level != INF

    def find_path(u):
        next_level = level[u] + 1
        for v in adj[u]:
            w = match_r[v]
            if w == -1:
                if next_level == free_level:
                    match_l[u] = v
                    match_r[v] = u
                    level[u] = INF
                    return True
            elif next_level < free_level and level[w] == next_level and find_path(w):
                match_l[u] = v
                match_r[v] = u
                level[u] = INF
                return True
        level[u] = INF
        return False

    while build_layers():
        for u in range(n):
            if match_l[u] == -1 and level[u] == 0:
                find_path(u)
    pairs = [(u, match_l[u]) for u in range(n) if match_l[u] != -1]
    return len(pairs), pairs


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, m, adj = read_bipartite(data)
    size, pairs = max_matching(n, m, adj)
    out = [str(size)]
    for u, v in pairs:
        out.append(f"{u} {v}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
```
