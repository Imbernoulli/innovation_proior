# Context

## Problem

Maintain a forest of $n$ nodes (each with a value) under online operations:
`link(u, v)` add an edge $(u, v)$ given they are in different trees; `cut(u, v)`
remove edge $(u, v)$; `connected(u, v)`; and path aggregate (e.g. sum or max of
node values) on the path between $u$ and $v$. $n$, ops up to $\sim 10^5$.

Each operation arrives online and must be answered before the next is read, so
the operation stream cannot be reordered or batched. Crucially the *shape* of the
forest changes over time: `link` and `cut` insert and delete edges, so the set of
trees and the paths inside them are not fixed in advance. The path between $u$ and
$v$ (when they are connected) is the unique simple path; it can contain anywhere
from one to $n$ nodes.

## Code framework

Each node carries an integer value; index $0$ is reserved as a null sentinel and
real nodes are numbered $1..n$. A generic balanced-BST `splay` primitive is
available as a building block: it stores a set of nodes in a binary search tree
and, given a node, rotates it to the root of its tree in amortized $O(\log n)$
while keeping the in-order sequence intact, so it can stand in for an ordered
sequence with fast splits and joins. What is missing is the structure that ties
such balanced-BST pieces to the changing forest, and the four top-level
operations.

```python
import sys


class BalancedBST:
    """Generic balanced binary search tree over node ids 1..n (0 is a null
    sentinel). `splay(x)` rotates x to the root of its tree in amortized
    O(log n), preserving the in-order order of the nodes. Each node stores a
    value and an aggregate over its subtree; children live in ch[x], the parent
    in fa[x]."""

    def __init__(self, n, values):
        self.ch = [[0, 0] for _ in range(n + 1)]  # [left, right]
        self.fa = [0] * (n + 1)
        self.val = [0] * (n + 1)
        self.agg = [0] * (n + 1)                   # aggregate over this subtree
        for i in range(1, n + 1):
            self.val[i] = self.agg[i] = values[i - 1]

    def _pushup(self, x):
        l, r = self.ch[x]
        self.agg[x] = self.agg[l] + self.val[x] + self.agg[r]

    def _side(self, x):
        return 1 if self.ch[self.fa[x]][1] == x else 0

    def _rotate(self, x):
        y = self.fa[x]
        z = self.fa[y]
        k = self._side(x)
        self.ch[z][self._side(y)] = x
        self.fa[x] = z
        w = self.ch[x][k ^ 1]
        self.ch[y][k] = w
        if w:
            self.fa[w] = y
        self.ch[x][k ^ 1] = y
        self.fa[y] = x
        self._pushup(y)
        self._pushup(x)

    def splay(self, x):
        # TODO: rotate x to the root of its tree (zig / zig-zig / zig-zag)
        pass


class Forest:
    """A dynamic forest with per-node values, built on top of the balanced-BST
    primitive. Node 0 is the null sentinel; real nodes are 1..n."""

    def __init__(self, n, values):
        # TODO: set up whatever per-node bookkeeping the forest needs
        pass

    def link(self, u, v):
        # TODO: add edge (u, v); u and v are in different trees
        pass

    def cut(self, u, v):
        # TODO: remove edge (u, v)
        pass

    def connected(self, u, v):
        # TODO: are u and v in the same tree?
        pass

    def path_query(self, u, v):
        # TODO: aggregate of values on the path between u and v
        pass


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    values = [int(next(it)) for _ in range(n)]
    forest = Forest(n, values)
    # (driver would read and dispatch the q operations here)


if __name__ == "__main__":
    main()
```
