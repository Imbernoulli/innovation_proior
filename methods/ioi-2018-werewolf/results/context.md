# Context

## Problem

You are given a connected undirected graph on $N$ vertices, numbered
$0, 1, \dots, N-1$, with $M$ edges. You must answer $Q$ independent queries.

Each query is four integers $(S, E, L, R)$ with $L \le R$. A query asks whether
there is a walk $v_0, v_1, \dots, v_k$ from $v_0 = S$ to $v_k = E$ that proceeds
in two phases separated by a single *switch* at some index $s$ (with
$0 \le s \le k$):

- Every vertex up to and including the switch is "big": $L \le v_0, v_1, \dots, v_s$.
- Every vertex from the switch onward is "small": $v_s, v_{s+1}, \dots, v_k \le R$.

The switch vertex $v_s$ therefore must satisfy $L \le v_s \le R$. Consecutive
vertices on the walk must be joined by an edge. The walk may revisit vertices.
Decide each query independently: output yes if such a walk exists, no otherwise.

Both $N$ and $Q$ can be large (each up to a few hundred thousand).

## Code framework

```python
import sys


class DSU:
    """Generic disjoint-set union over a fixed universe, with path compression.
    union(a, b) merges the sets; find(x) returns a set representative."""

    def __init__(self, size):
        self.par = list(range(size))

    def find(self, x):
        root = x
        while self.par[root] != root:
            root = self.par[root]
        while self.par[x] != root:
            self.par[x], x = root, self.par[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.par[rb] = ra
        return ra


def answer_queries(n, edges, queries):
    """edges: list of (u, v), 0-based undirected. queries: list of (S, E, L, R)
    with L <= R. For each query, return whether a walk from S to E exists that
    stays on vertices >= L up to a single switch vertex v (with L <= v <= R),
    then stays on vertices <= R. Returns a list of bools, one per query."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    def build_structure(order):
        # TODO
        return None

    def make_jumps(parent):
        # TODO
        return None

    def locate_first(S, L):
        # TODO
        return None

    def locate_second(E, R):
        # TODO
        return None

    def add_value(i):
        # TODO
        pass

    def prefix_value(i):
        # TODO
        return 0

    # TODO

    return [False] * len(queries)


if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); q = int(next(it))
    edges = [(int(next(it)), int(next(it))) for _ in range(m)]
    queries = [(int(next(it)), int(next(it)), int(next(it)), int(next(it)))
               for _ in range(q)]
    out = answer_queries(n, edges, queries)
    print("\n".join("YES" if a else "NO" for a in out))
```
