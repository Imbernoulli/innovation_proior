# Context

## Problem

Given a directed acyclic graph on $n$ vertices, find the minimum number of
vertex-disjoint paths needed to cover every vertex exactly once (the minimum path
cover). A path here is a directed simple path that follows edges of the graph; a
single vertex with no edges counts as a path of length zero. "Vertex-disjoint" and
"exactly once" mean the chosen paths partition the vertex set: every vertex lies on
one and only one path. The goal is to make the number of paths as small as
possible.

The graph is acyclic, so the edge relation is consistent with some topological
order, and no path can revisit a vertex. Sizes are moderate: $n$ up to a few
thousand vertices and $m$ up to tens of thousands of edges, small enough that an
$O(n \cdot m)$ procedure is acceptable.

## Code framework

The graph is read into a $0$-based adjacency list of directed edges. The helper
functions and input/output wrapper are fixed. What is missing is the top-level
`min_path_cover(n, adj)` that turns the graph into the integer answer.

```python
import sys

sys.setrecursionlimit(1_000_000)


def read_dag(data):
    """Parse n, m, and m directed edges (1-based in input) into a 0-based
    adjacency list of a DAG. Returns (n, adj)."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(m):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
    return n, adj


def bipartite_matching(n_left, n_right, adj):
    """Maximum-cardinality matching of a bipartite graph by Kuhn's
    augmenting-path search. adj[u] lists right vertices joined to left vertex u.
    Returns the size of a maximum matching. O(V * E)."""
    match_right = [-1] * n_right       # right vertex -> its matched left vertex

    def try_kuhn(u, used):
        for w in adj[u]:
            if not used[w]:
                used[w] = True
                if match_right[w] == -1 or try_kuhn(match_right[w], used):
                    match_right[w] = u
                    return True
        return False

    size = 0
    for u in range(n_left):
        used = [False] * n_right
        if try_kuhn(u, used):
            size += 1
    return size


def min_path_cover(n, adj):
    # TODO
    pass


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, adj = read_dag(data)
    print(min_path_cover(n, adj))


if __name__ == "__main__":
    main()
```
