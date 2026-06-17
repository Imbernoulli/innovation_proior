## Problem

You are given a tree on $n$ vertices. Each edge has a non-negative integer
length, and you are given an integer $K$.

Among all simple paths between two distinct vertices, consider those whose total
edge length is exactly $K$. Output the minimum possible number of edges on such
a path, or $-1$ if no path has total length exactly $K$.

## Code framework

```python
import sys

def race(n, K, edges):
    """edges: list of (u, v, w) with 0-based vertices and non-negative w.
    Returns the minimum number of edges on a path of total length exactly K,
    or -1 if no such path exists."""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    removed = [False] * n
    size = [0] * n
    parent = [-1] * n

    best_depth = [0] * (K + 1)
    seen = [-1] * (K + 1)
    stamp = 0
    answer = -1

    def calc_size(root):
        # TODO
        pass

    def find_centroid(order, parent, total):
        # TODO
        pass

    def dfs_collect(start, c0, centroid):
        # TODO
        pass

    def process(start):
        # TODO
        pass

    sys.setrecursionlimit(1 << 20)
    process(0)
    return answer


if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()
    if data:
        it = iter(data)
        n = int(next(it)); K = int(next(it))
        edges = [(int(next(it)), int(next(it)), int(next(it)))
                 for _ in range(n - 1)]
        print(race(n, K, edges))
```
