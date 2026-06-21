# Context

## Problem

You are given a rooted tree of `n` nodes, rooted at node `0`. Each node has a
color. For every node `v`, look at the multiset of colors in the subtree of
`v`. A color is dominating in that subtree if no other color appears more times
there. If several colors tie for the maximum frequency, all of them are
dominating.

Return one value for every node: the sum of all dominating colors in that
node's subtree.

The input size can be as large as `n = 100000`, and every color is an integer in
`[1, n]`.

## Code framework

The tree is passed as an undirected edge list. `color[v]` is the color of node
`v`, using zero-based node indices. The required artifact is a single top-level
`solve` function that returns the answer list indexed by node. The input wrapper
uses one-based input vertices and converts them before calling `solve`.

```python
import sys
from sys import setrecursionlimit


def solve(n, color, edges):
    """color[v] is the color of node v (0-based); edges is a list of (a, b)
    undirected tree edges. Return ans where ans[v] is the sum of all
    dominating colors in the subtree of v (tree rooted at node 0)."""
    g = [[] for _ in range(n)]
    for a, b in edges:
        g[a].append(b)
        g[b].append(a)

    ans = [0] * n

    # TODO

    return ans


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    color = [int(next(it)) for _ in range(n)]
    edges = []
    for _ in range(n - 1):
        x = int(next(it)) - 1
        y = int(next(it)) - 1
        edges.append((x, y))
    ans = solve(n, color, edges)
    sys.stdout.write(" ".join(map(str, ans)) + "\n")


if __name__ == "__main__":
    main()
```
