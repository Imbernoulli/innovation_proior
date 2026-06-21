# Context

## Problem

Given a directed graph (flowgraph) with a start vertex $s$, a vertex $u$ dominates $v$ if every path from $s$ to $v$ passes through $u$. For every vertex $v$ reachable from $s$, compute its immediate dominator $\mathrm{idom}(v)$ (the unique dominator of $v$, other than $v$, that is dominated by all other dominators of $v$). Output the dominator tree ($\mathrm{parent}[v] = \mathrm{idom}(v)$).

The graph may contain cycles, self-loops, and parallel edges, and not every vertex need be reachable from $s$ — vertices unreachable from $s$ have no dominators to report and are left out of the tree. The graph can be large ($n$ vertices and $m$ edges, each up to $\sim 10^5$ or more).

## Code framework

The graph is read into a $0$-based successor adjacency list; the predecessor adjacency list is the reverse. The missing piece is the top-level routine that consumes the graph and returns, for every reachable vertex, its immediate dominator.

```python
import sys


def read_graph(data):
    """Parse n, m, the start s (1-based in input), and m directed edges into a
    0-based successor list and its reverse. Returns (n, s, succ, pred)."""
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    s = int(next(it)) - 1
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for _ in range(m):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        succ[a].append(b)
        pred[b].append(a)
    return n, s, succ, pred


def dominator_tree(n, s, edges):
    """idom[v] = the immediate dominator of v for every v reachable from s;
    idom[s] = s by convention; idom[v] = -1 for v unreachable from s."""
    succ = [[] for _ in range(n)]
    pred = [[] for _ in range(n)]
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    idom = [-1] * n
    # TODO
    return idom


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, s, succ, pred = read_graph(data)
    edges = [(a, b) for a in range(n) for b in succ[a]]
    idom = dominator_tree(n, s, edges)
    # (driver would print the dominator tree / per-vertex idom here)


if __name__ == "__main__":
    main()
```
