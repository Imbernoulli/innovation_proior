The object in front of me is a rooted tree on `n` nodes, rooted at `0`, and a stream of queries. Each query gives two nodes `u` and `v` and asks for their lowest common ancestor: the deepest node that is an ancestor of both. The direct way to answer one query is to keep parent pointers and depths, move the deeper node upward until both nodes have the same depth, then move both upward together until they meet. That is correct, but on a path-shaped tree the query costs `Theta(n)`, and repeating that for many queries spends all the time walking over the same chains again and again.

The first improvement is to jump upward by powers of two. If I precompute `up[k][v]`, the `2^k`-th ancestor of `v`, then I can equalize depths by decomposing the depth difference into binary, and I can lift both nodes from the largest power down while keeping them below the answer. The preprocessing is `O(n log n)`, and each query costs `O(log n)`. That is much better, but the query still climbs the tree. If I want `O(1)` per query, the query cannot do a logarithmic climb; it has to become a fixed number of table lookups.

A tree is awkward for constant-time lookup, but an immutable array is friendly. So I need a way to flatten the rooted tree into an array without losing the ancestor information that the query needs. Recording each node only when I first enter it gives preorder, but preorder hides the returns. If two nodes are in different child subtrees, the important action is not just that I entered each subtree; it is that the traversal climbed back up through their common ancestors. Preorder throws that away. I should record the walk itself: append a node when I first enter it, and append it again each time I return to it from a child.

On a small tree, root `1` with children `2`, `3`, `4`, node `2` with children `5`, `6`, and node `4` with child `7`, that walk records
`1, 2, 5, 2, 6, 2, 1, 3, 1, 4, 7, 4, 1`.
Every edge is crossed once downward and once upward, and the root is recorded at the start, so the length is `2n - 1`. This is still linear, so it is affordable to keep.

Now I need to see where the answer sits in this array. Take nodes `6` and `4` in that example. Their first appearances are at positions `4` and `9`. The slice between those first appearances is `6, 2, 1, 3, 1, 4`. The LCA is `1`, and it appears there because the traversal cannot move from the first side of the tree to the second side without climbing through the lowest common ancestor. It also does not need to climb above that ancestor before the second node's first appearance. Any extra work in the slice is a detour into a child subtree, and a detour only goes deeper than the node it leaves from.

That depth observation singles out the answer. In the slice between the first appearances of `u` and `v`, the LCA is the shallowest node. Nodes on the path below it are deeper, and side-subtree nodes are deeper still. If the same shallowest node appears more than once in the slice, those positions are still the same LCA node, so any minimum-depth position is safe. For the example, the depths in the slice from `6` to `4` are `3, 2, 1, 2, 1, 2`, and each minimum position holds node `1`.

So I keep three arrays from the traversal: `euler[pos]` is the node touched at that position, `depth_at[pos]` is its depth, and `first[node]` is the first position where that node appears. For a query I take `left = first[u]` and `right = first[v]`, swap them if needed, and the answer is `euler[pos]`, where `pos` is a position of minimum `depth_at` in the inclusive range `[left, right]`. The tree problem has become a static range-minimum query over the depth array. The important part is that the minimum key is the depth, not the Euler index. Choosing the smallest Euler index in the range would just return the left endpoint; choosing the minimum depth returns the ancestor reached by the climb.

Now the array primitive fits perfectly. For a static range minimum, I can build a sparse table. At level `k`, table entry `table[k][start]` stores the position of the minimum key in the block of length `2^k` starting at `start`. The base level stores the positions themselves. A block of length `2^k` is two adjacent blocks of length `2^(k-1)`, so I compare the two stored positions from the previous level and keep the one whose key is smaller. This stores positions, not values, because after the range minimum I still need to recover the node from `euler[pos]`.

For a query range `[left, right]`, let `length = right - left + 1` and `k = floor(log2(length))`. The two blocks `[left, left + 2^k - 1]` and `[right - 2^k + 1, right]` cover the whole range. They may overlap, but minimum is idempotent: seeing the same candidate twice cannot change the answer. So the range minimum is just the better of two stored positions. If I precompute the floor-log table for all lengths, the query has no loop.

The traversal should be iterative. A recursive DFS is elegant, but a path-shaped tree can have height `n`, and then recursion depth becomes a separate failure mode. I can simulate DFS with stack frames `(node, depth, next_child_index)`. When I first push a child, I append that child and its depth and set its first position. When a frame is exhausted, I pop it; if there is still a parent frame underneath, the walk has just returned to that parent, so I append the parent and the parent's depth. Marking a node as seen on entry is enough to handle the undirected adjacency list.

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

The result is exactly the shape I wanted: one traversal creates a length-`2n - 1` Euler array, a first-position array, and a depth array; the sparse table preprocesses the depth array in `O(n log n)` while storing minimizing positions; each query converts the two nodes to a first-occurrence interval, takes the minimum-depth position in that interval with two table lookups, and returns the Euler node at that position in `O(1)`.
