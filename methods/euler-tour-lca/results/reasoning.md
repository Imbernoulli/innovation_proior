The object in front of me is a rooted tree on `n` nodes, rooted at `0`, and a stream of queries. Each query gives two nodes `u` and `v` and asks for their lowest common ancestor: the deepest node that is an ancestor of both. The direct way to answer one query is to keep parent pointers and depths, move the deeper node upward until both nodes have the same depth, then move both upward together until they meet. That is correct, but on a path-shaped tree the query costs `Theta(n)`, and repeating that for many queries spends all the time walking over the same chains again and again.

The first improvement is to jump upward by powers of two. If I precompute `up[k][v]`, the `2^k`-th ancestor of `v`, then I can equalize depths by decomposing the depth difference into binary, and I can lift both nodes from the largest power down while keeping them below the answer. The preprocessing is `O(n log n)`, and each query costs `O(log n)`. That is much better, but the query still climbs the tree. If I want `O(1)` per query, the query cannot do a logarithmic climb; it has to become a fixed number of table lookups.

A tree is awkward for constant-time lookup, but an immutable array is friendly. So I need a way to flatten the rooted tree into an array without losing the ancestor information that the query needs. Recording each node only when I first enter it gives preorder, but preorder hides the returns. If two nodes are in different child subtrees, the important action is not just that I entered each subtree; it is that the traversal climbed back up through their common ancestors. Preorder throws that away. I should record the walk itself: append a node when I first enter it, and append it again each time I return to it from a child.

Let me make that concrete on a small tree so I can see what the array actually looks like. Root `1` has children `2`, `3`, `4`; node `2` has children `5`, `6`; node `4` has child `7`. Walking it and writing down every touch, I get

`1, 2, 5, 2, 6, 2, 1, 3, 1, 4, 7, 4, 1`.

I should sanity-check the length rather than take it on faith. There are `n = 7` nodes; every edge is crossed once on the way down and once on the way back, so the number of touches after the initial root touch is `2 * (n - 1) = 12`, plus the root itself, giving `13`. Counting the array above: `1 2 5 2 6 2 1 3 1 4 7 4 1` is indeed 13 entries, and `2n - 1 = 13`. So the array is linear in `n`, which is affordable to keep.

Now I need to see where the answer sits in this array. Take nodes `6` and `4`. Their first appearances are at positions `4` and `9` (counting from `0`). The slice between those first appearances is `6, 2, 1, 3, 1, 4`. The LCA of `6` and `4` is the root `1`, and `1` does appear inside that slice. Why it must appear is the part I want to be sure of: the traversal cannot move from the first node's subtree to the second node's subtree without passing through every common ancestor of the two, and the lowest of those is the LCA. It also does not climb above the LCA before reaching the second node's first appearance, because the second node still lives below the LCA. Any other node in the slice is either on the downward path past the LCA or a detour into a side subtree, and both of those go strictly deeper than the LCA.

That last sentence is really a claim about depth, so let me read off the depths along the slice and check it. With the root at depth `0`, the depths of `6, 2, 1, 3, 1, 4` are `2, 1, 0, 1, 0, 1`. The smallest value is `0`, attained at the two positions that hold node `1`, and `1` is exactly the LCA. The deepest-ancestor node is the shallowest node in the slice — which matches the picture above. (I'll note the depth uses a root depth of `0`; if I had instead started depths at `1`, the same slice would read `3, 2, 1, 2, 1, 2`, and the argument is unchanged, but I should keep the convention consistent with the code, which sets the root to `0`.)

So in the slice between the first appearances of `u` and `v`, the LCA is a node of minimum depth. If that shallowest node appears more than once in the slice, every such position still holds the same node, so returning any minimum-depth position is safe. This turns the tree query into a static range-minimum query over a depth array, with the catch that the key being minimized must be the depth, not the array index. To make sure I have that distinction right, I can ask what would happen if I minimized the Euler index instead: the smallest index in `[left, right]` is just `left`, so it would always return `euler[left]`, i.e. whichever of `u`, `v` appears first. Checking the example, for `u = 6, v = 4` that would return `6`, which is wrong. Minimizing depth is what reaches the ancestor.

Before committing to this I want to confirm it on the whole tree, not only the one pair I picked, because it would be easy for the depth argument to miss a case — for instance when one node is an ancestor of the other, where the slice does not straddle the LCA in the obvious way. So I take the same tree and, for every ordered pair of nodes, compare the Euler-tour-plus-range-minimum answer against a brute-force LCA computed by walking parent pointers up to a meeting point. Over all `7 * 7 = 49` pairs the two agree on every one; there are no mismatches. The ancestor case is worth looking at directly: for `u = 2` (an ancestor of `v = 6`), the first positions are `1` and `4`, the slice is `2, 5, 2, 6` with depths `1, 2, 1, 2`, the minimum depth `1` lands on node `2`, and that is the correct answer. So the construction handles both the straddling case and the ancestor case, and the brute-force agreement gives me reasonable confidence the slice-minimum rule is correct in general, not just on the pair I happened to draw.

So I keep three arrays from the traversal: `euler[pos]` is the node touched at that position, `depth_at[pos]` is its depth, and `first[node]` is the first position where that node appears. For a query I take `left = first[u]` and `right = first[v]`, swap them if needed, and the answer is `euler[pos]`, where `pos` is a position of minimum `depth_at` in the inclusive range `[left, right]`.

Now the array primitive fits the static-range-minimum job. I can build a sparse table. At level `k`, table entry `table[k][start]` stores the position of the minimum key in the block of length `2^k` starting at `start`. The base level stores the positions themselves. A block of length `2^k` is two adjacent blocks of length `2^(k-1)`, so I compare the two stored positions from the previous level and keep the one whose key is smaller. The table stores positions rather than values, because after the range minimum I still need to recover the node from `euler[pos]`.

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

Stepping back, the pieces line up the way I needed them to. One traversal creates a length-`2n - 1` Euler array, a first-position array, and a depth array; the sparse table preprocesses the depth array in `O(n log n)` while storing minimizing positions; each query converts the two nodes to a first-occurrence interval, takes the minimum-depth position in that interval with two table lookups, and returns the Euler node at that position in `O(1)`. The all-pairs check on the small tree, including the ancestor case, is what convinces me the depth-minimum rule is doing the right thing rather than merely looking plausible.
