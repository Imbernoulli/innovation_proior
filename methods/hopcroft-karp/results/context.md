# Context

## Problem

Given a bipartite graph with left part `L` of size `n`, right part `R` of size `m`, and edges only between `L` and `R`, find a maximum-cardinality matching: the largest possible set of edges such that no two chosen edges share an endpoint.

The graph is supplied as a `0`-based adjacency list `adj`, where `adj[u]` contains the right-side vertices adjacent to left vertex `u`. A valid output can be represented by two partner arrays: `match_l[u]` is the right vertex paired with left vertex `u`, or `-1` if `u` is unmatched; `match_r[v]` is the left vertex paired with right vertex `v`, or `-1` if `v` is unmatched. The required result is the matching size and the list of matched `(left, right)` pairs.

## Code framework

The parser and output harness are already fixed. The missing work is the search inside `max_matching` that fills `match_l` and `match_r` with a largest valid set of pairs.

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
    # TODO: fill match_l and match_r with a largest valid set of pairs.
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
