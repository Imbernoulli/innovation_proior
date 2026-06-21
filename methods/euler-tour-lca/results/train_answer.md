We are given a rooted tree on $n$ nodes and a stream of $q$ queries, each asking for the lowest common ancestor of two nodes $u$ and $v$ — the deepest node that is an ancestor of both. The naive answer keeps parent pointers and depths, lifts the deeper node until both share a depth, then walks both up together until they collide; this is correct but costs $\Theta(n)$ per query on a path-shaped tree, so it re-walks the same chains over and over. Binary lifting fixes the asymptotics part way: precompute $\mathrm{up}[k][v]$, the $2^k$-th ancestor of $v$, equalize depths by reading the binary expansion of the depth difference, and lift both nodes from the largest power downward while keeping them strictly below the answer. That gives $O(n \log n)$ preprocessing and $O(\log n)$ per query. But the query still climbs the tree, and the requirement is $O(1)$ per query — which means a query cannot afford any logarithmic climb at all. It has to collapse into a fixed number of array reads.

The obstacle is that a tree is awkward for constant-time lookup while an immutable array is friendly, so I propose to flatten the tree into an array without losing the ancestor information the query needs, and then answer each query as a static range-minimum. The method is LCA by Euler tour with sparse-table range-minimum. The flattening must record not just where the traversal enters each node but where it climbs back out, because the entire point of an LCA query for two nodes in different child subtrees is the moment the walk climbs through their common ancestor — and preorder, which records a node only on first entry, throws exactly that away. So I record the walk itself: append a node when the depth-first traversal first enters it, and append it again every time the traversal returns to it from a finished child. On the tree rooted at $1$ with children $2,3,4$, where $2$ has children $5,6$ and $4$ has child $7$, this walk is $1, 2, 5, 2, 6, 2, 1, 3, 1, 4, 7, 4, 1$. Every edge is crossed once down and once up and the root is written once at the start, so the array has length exactly $2n-1$, which is linear and therefore affordable to keep.

What makes this array carry the answer is a depth argument. Take two nodes and look at the slice of the Euler array between their first appearances; sort the endpoints so $\mathit{left} = \mathrm{first}[u]$ and $\mathit{right} = \mathrm{first}[v]$ with $\mathit{left} \le \mathit{right}$. Between the first appearance of one node and the first appearance of the other, the traversal must move from the first node's side of the tree up through the lowest common ancestor and then down toward the second node's side; it never has to climb above that ancestor before reaching the second node, and any other work inside the slice is a detour into a child subtree, which only ever goes deeper than the node it departs from. Therefore the lowest common ancestor is precisely the shallowest node in that slice, and the answer is the Euler entry at a minimum-depth position:

$$
\mathrm{LCA}(u, v) =
\mathrm{euler}\!\left[\, \arg\min_{\mathit{left} \le i \le \mathit{right}} \mathrm{depth\_at}[i] \,\right].
$$

The load-bearing choice here is that the key being minimized is the depth, not the Euler index. If I minimized the index I would just get the left endpoint back; minimizing depth returns the node the climb actually reaches. If the shallowest node happens to appear several times in the slice, every one of those positions is the same ancestor node, so taking any minimum-depth position is safe. To support this I keep three arrays from the single traversal: $\mathrm{euler}[\mathit{pos}]$ is the node touched at that position, $\mathrm{depth\_at}[\mathit{pos}]$ is its depth, and $\mathrm{first}[\mathit{node}]$ is the first position where that node appears.

The tree problem is now a static range minimum over the depth array, and for that I build a sparse table. At level $k$, the entry $\mathrm{table}[k][\mathit{start}]$ stores the *position* of the minimum-depth element in the block of length $2^k$ beginning at $\mathit{start}$; the base level stores the positions themselves, $0,1,\dots,m-1$. A block of length $2^k$ is two adjacent blocks of length $2^{k-1}$, so I compare the two positions stored at the previous level and keep the one whose depth is smaller — that is the entire build recurrence, and it costs $O(m \log m)$ for $m = 2n-1$. The table stores positions rather than depth values precisely because, after the range minimum, I still need to recover the node through $\mathrm{euler}[\mathit{pos}]$. To answer a range $[\mathit{left}, \mathit{right}]$ in constant time, let $\mathit{length} = \mathit{right} - \mathit{left} + 1$ and $k = \lfloor \log_2 \mathit{length} \rfloor$; the two blocks $[\mathit{left}, \mathit{left} + 2^k - 1]$ and $[\mathit{right} - 2^k + 1, \mathit{right}]$ together cover the whole range. They may overlap, but minimum is idempotent — seeing the same candidate twice never changes the result — so the answer is simply the better of two stored positions, with the floor-log table precomputed so the query contains no loop.

One implementation choice in the traversal matters: it must be iterative. A recursive depth-first search is elegant, but a path-shaped tree has height $n$, and then recursion depth itself becomes a separate failure mode. I simulate the search with explicit stack frames $(\mathit{node}, \mathit{depth}, \mathit{child\_index})$. When the search descends to a new child I append that child and its depth and set its first position; when a frame's children are exhausted I pop it, and if a parent frame remains underneath, the walk has just returned to that parent, so I append the parent again at the parent's depth. Marking each node as seen on entry is enough to handle the undirected adjacency list without revisiting. The result is exactly the intended shape: one traversal produces the length-$2n-1$ Euler array, the first-position array, and the depth array; the sparse table preprocesses the depth array in $O(n \log n)$ time and space while storing minimizing positions; and each query turns its two nodes into a first-occurrence interval, takes the minimum-depth position with two table reads, and returns the Euler node there in $O(1)$.

```python
import sys


def read_input(data):
    """Parse n, q, n-1 edges, and q queries from 1-based input."""
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    queries = []
    for _ in range(q):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        queries.append((u, v))
    return n, adj, queries


class SparseTableMin:
    """Range minimum over a fixed integer array.

    Each table cell stores the position whose key is minimal, so a caller can
    recover the value associated with that position. Build is O(m log m);
    queries are O(1).
    """

    def __init__(self, key):
        self.key = key
        m = len(key)
        self.log = [0] * (m + 1)
        for i in range(2, m + 1):
            self.log[i] = self.log[i >> 1] + 1

        levels = self.log[m] + 1 if m else 1
        table = [[0] * m for _ in range(levels)]
        table[0] = list(range(m))
        for k in range(1, levels):
            half = 1 << (k - 1)
            span = half << 1
            row = table[k]
            prev = table[k - 1]
            for start in range(m - span + 1):
                left = prev[start]
                right = prev[start + half]
                row[start] = left if key[left] <= key[right] else right
        self.table = table

    def argmin(self, left, right):
        """Return a position of the minimum key in inclusive range [left, right]."""
        length = right - left + 1
        k = self.log[length]
        a = self.table[k][left]
        b = self.table[k][right - (1 << k) + 1]
        return a if self.key[a] <= self.key[b] else b


class LCA:
    def __init__(self, n, adj, root=0):
        self.n = n
        self.adj = adj
        self.preprocess(root)

    def preprocess(self, root):
        euler = []
        depth_at = []
        first = [-1] * self.n
        seen = [False] * self.n

        stack = [(root, 0, 0)]
        seen[root] = True
        first[root] = 0
        euler.append(root)
        depth_at.append(0)

        while stack:
            node, depth, child_index = stack[-1]
            children = self.adj[node]
            advanced = False

            while child_index < len(children):
                nxt = children[child_index]
                child_index += 1
                if seen[nxt]:
                    continue

                stack[-1] = (node, depth, child_index)
                seen[nxt] = True
                first[nxt] = len(euler)
                euler.append(nxt)
                depth_at.append(depth + 1)
                stack.append((nxt, depth + 1, 0))
                advanced = True
                break

            if not advanced:
                stack.pop()
                if stack:
                    parent, parent_depth, _ = stack[-1]
                    euler.append(parent)
                    depth_at.append(parent_depth)

        self.euler = euler
        self.first = first
        self.depth_at = depth_at
        self.rmq = SparseTableMin(depth_at)

    def lca(self, u, v):
        left = self.first[u]
        right = self.first[v]
        if left > right:
            left, right = right, left
        pos = self.rmq.argmin(left, right)
        return self.euler[pos]


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, adj, queries = read_input(data)
    solver = LCA(n, adj)
    out = [str(solver.lca(u, v) + 1) for u, v in queries]
    sys.stdout.write("\n".join(out))


if __name__ == "__main__":
    main()
```
