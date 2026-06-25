In front of me is a tree on $n$ nodes, each node carrying a value, and two operations I have to answer one after another, online: set the value at a node, and report the aggregate — sum, or maximum; let me hold sum in my head and keep the aggregate swappable — of the values on the path between two nodes $u$ and $v$. Think $n$ and the number of operations both up to about $10^5$. The path between $u$ and $v$ is the unique simple path in the tree. Let me start from the only honest thing: I can walk that path. Find the path from $u$ to $v$, add up the node values, done.

So cost it. To even find the path I'd do a search from $u$ until I reach $v$ and read off ancestors, or root the tree and walk both endpoints up to their meeting point. Either way the path itself can be long — on a bamboo (a single chain of $n$ nodes) the path from one end to the other is all $n$ nodes. So one query is $O(n)$, and with $q$ queries that's $O(nq)$, around $10^{10}$. Dead. And updates being interleaved means I can't precompute all answers either. The thing killing me is that a single path can sweep through $\Theta(n)$ nodes, and I'm re-summing them from scratch every time.

When a query is over a contiguous *range* of an array, I know exactly what to reach for — a segment tree: point update in $O(\log n)$, range aggregate in $O(\log n)$. The path values are a multiset I want to aggregate; if only they sat in a contiguous slice of some array, I'd be done in $O(\log n)$ per query and per update. So the real question isn't "how do I sum a path," it's "can I lay the tree's nodes into a linear array so that the nodes of a path form a small number of contiguous ranges?"

Let me see how far an arbitrary linear order gets me. Root the tree and write down each node's position in DFS-visit order — the standard Euler/DFS index, `pos[v]`. This has one beautiful property: the nodes of any *subtree* occupy a contiguous block of `pos`, because DFS enters a subtree, finishes all of it, and only then leaves. So subtree-aggregate is one range query. But a *path* between $u$ and $v$ is not a subtree. Split the path at the lowest common ancestor $l$ of $u$ and $v$: it's the upward path $u \to l$ glued to the upward path $v \to l$. Each of those is a vertical chain of ancestors. And in plain DFS order, the ancestors of a node are scattered all over `pos` — the parent of $v$ sits at the front of $v$'s subtree block, its grandparent further out, and so on; an ancestor chain is in general $\Theta(\text{depth})$ separate positions, not a contiguous run. So plain DFS order gives me subtrees for free but does nothing for vertical paths. On a bamboo the ancestor chain is the whole array and it's contiguous by luck, but on a bushy tree the chain is shredded. I need a *different* linear order, one tuned so that the vertical chains I actually query are contiguous.

Back up and think about what a path query really costs if I'm willing to use many ranges. Suppose I had a separate segment tree for every root-to-leaf chain I could decompose the tree into. Then $u \to l$ would cross some number of these chains, and on each crossed chain I'd do one range query. The total cost per query is (number of chains crossed) $\times$ ($\log n$ for the range query on each). So everything hinges on the *number of chains a single root-to-node path can cross*. If I cut the tree into chains naively — say, peel off arbitrary root-to-leaf paths — an adversary builds a "broom": a long handle, then a fan of leaves at the bottom. Cut along one handle-to-leaf path, and every *other* leaf is its own chain of length one, and a path from the root down to... no, wait, that broom is fine for a single descent. Let me find the actual adversary. A complete binary tree: any single root-to-node path has length $\Theta(\log n)$, and if I cut greedily I might still cross $\Theta(\log n)$ chains — that's fine. The dangerous shape is the "caterpillar" or a star of long chains: pick chains badly and a root-to-node path zig-zags, entering and leaving chain after chain, $\Theta(n)$ of them. So an *arbitrary* chain decomposition gives no bound at all. The number of chains crossed is entirely at the mercy of *which* edges I declare to be "within a chain" versus "between chains." I get to choose the cut. I want to choose it so that no root-to-node path ever crosses more than $O(\log n)$ chain boundaries.

So at every node with several children, exactly one child's edge will be "stay on the same chain" and the others will be "start a new chain." Crossing into one of those other children is a chain boundary. I want any descent from the root to cross few boundaries. A descent crosses a boundary precisely when it steps onto a node via a "new-chain" edge. So I want to arrange that *most* steps of any descent are the "same-chain" kind, and the "new-chain" kind is rare on every path — at most $O(\log n)$ of them, no matter the path.

What makes a step rare-able? I get to pick, per node, which one child continues the chain. Picking blindly fails: on a star-of-chains, if at the center I continue into chain $A$, then descending into chain $B$ costs a boundary, but that's just one boundary per descent, fine — the real trouble is nested. Picture a path graph where I keep making the *wrong* choice: a node has two children, I continue into the small side, and the descent that actually matters goes into the big side, paying a boundary, then again, then again. So the choice has to be tied to *subtree size*. Let me make the continuing child the one with the **largest subtree**. Then think about what happens when a descent takes a "new-chain" edge — an edge into a child that is *not* the largest. Call the node $v$ with subtree size $s(v)$, and the child I descend into $c$, with $c$ not the largest child. The largest child has subtree at least as big as $c$'s, and both are disjoint pieces of $v$'s subtree, so $s(v) \ge s(c) + s(\text{largest child}) \ge 2\,s(c)$, giving

$$s(c) \le \frac{s(v)}{2}.$$

Every time a descent takes one of these "new-chain" edges — a step into a non-largest child — the subtree size of where I'm standing at least halves. The subtree size starts at $n$ (root) and can't drop below $1$, so a descent can take at most $\log_2 n$ such steps before the subtree would shrink below one node, which is impossible. So if I name the edge to the largest-subtree child the *heavy* edge and the rest *light* edges, and call a maximal run of consecutive heavy edges one chain, then any root-to-node path crosses a light edge — equivalently switches chains — at most $\log_2 n$ times, putting it on at most $O(\log n)$ chains. Notice this argument never mentioned the tree's shape; the halving comes purely from "$c$ is not the biggest child," so the broom, the caterpillar, the star-of-chains all obey the same $\log_2 n$ ceiling. Let me put a number on it to be sure I'm not fooling myself: at $n = 10^5$, $\log_2 n \approx 16.6$, so the worst descent touches at most $17$ chains, not the $10^5$ that the naive cut allowed. That is the gap I needed.

Let me sanity-check the "one continuing child" is well defined and that chains tile the tree. At each node I pick exactly one heavy child (the largest subtree; ties broken arbitrarily — if there are two equal-largest children, only one gets the heavy edge, and the halving still holds because the *other* equal one is then a non-largest child with subtree $\le s(v)/2$). Could a node be the heavy child of two parents? No — it has one parent. So the heavy edges form a set where each node has at most one heavy edge going down (its chosen heavy child) and at most one heavy edge coming in (it is the heavy child of at most its single parent). A graph where every vertex has in-degree and out-degree at most one in these edges is a disjoint union of simple paths — the heavy edges decompose into disjoint top-to-bottom runs, and every node lies on exactly one such chain (a node with no incoming heavy edge is the head; a node with no heavy child is a chain of length one). So the chains partition the vertices: disjoint and covering.

Now the part I actually care about for the array: I want each chain to be a *contiguous* block of positions, so a query along (part of) a chain is one segment-tree range. This is where the DFS order I dismissed earlier comes back, but with a twist. Do the DFS, and at each node visit the **heavy child first**, before any light children. Then when I enter a node and walk down its heavy child, and that node's heavy child, and so on, I'm walking straight down a heavy chain, assigning consecutive `pos` values the whole way before I ever back up to take a light branch. So an entire heavy chain gets a run of consecutive `pos` values — top of the chain smallest, bottom largest. Each chain is now an interval `[pos[head], pos[bottom]]` of the array, where `head` is the topmost node of the chain. That's exactly the contiguity I need. And I'll store `head[v]` = the top node of $v$'s chain, so from any node I can see which chain it's on and jump to the top.

So `pos` lays all nodes into one flat array, chains are contiguous in it, and one segment tree over that array handles point updates (set value at `pos[u]`) and range aggregates (over any sub-interval, in particular any prefix-or-segment of a chain). One tree, not one-per-chain — the chains partition the nodes, so the array has exactly $n$ slots and a single $O(n)$ segment tree covers all of them.

Now answering `path_query(u, v)` by climbing chain by chain. The path is $u \to l \to v$. I don't even want to compute $l$ separately if I can avoid it — let me see if the climb finds it for free. Keep two pointers $u$ and $v$. If they're on the same chain (`head[u] == head[v]`), then the path between them is the slice of that one chain between their two positions, one range query, and I'm done. Otherwise they're on different chains. I want to lift one of them up a whole chain at a time. Which one? Lift the one whose chain *head is deeper*, because that node's chain head, being deeper, cannot be an ancestor of the other pointer — so the segment of the path on that chain, from the chain head down to the pointer, definitely belongs to the answer and can be aggregated now, and then I jump the pointer to the parent of its chain head, stepping up one light edge onto the next chain. Concretely: if `depth[head[u]] < depth[head[v]]` swap $u, v$ so $u$ is the one with the deeper head; aggregate the chain slice `query(pos[head[u]], pos[u])`; set `u = parent[head[u]]`. Repeat. Each iteration moves one pointer up one whole chain, i.e. across one light edge, and I argued a root-to-node path has $O(\log n)$ of those — so this loop runs $O(\log n)$ times. When the heads finally coincide, the two pointers are on the same chain, the deeper of them descended-from the shallower; the shallower one is the lowest common ancestor $l$, and the final `query` over `[pos[shallower], pos[deeper]]` adds the last stretch through $l$. The LCA dropped out of the climb — I never computed it on the side.

I'm wary of an off-by-one here — double-counting the meeting node, or skipping the join. The structure of the argument is: while heads differ I aggregate `[pos[head[u]], pos[u]]` and then jump to `parent[head[u]]`, which is strictly above the chain I just consumed — so no overlap with what's left, and no gap either, because the parent of the chain head is the very next node up the path. When heads match I take `[pos[min], pos[max]]` inclusive, which should include $l$ exactly once and every node from $l$ down to the deeper pointer. But "should" is exactly the kind of claim I keep getting wrong by one, so let me not trust the prose and instead run the whole thing on a concrete tree.

Take this tree, rooted at $1$, with node values $10,20,\dots,70$:

```
            1 (10)
           / \
          2   3
        (20) (30)
        / \    \
       4   5    6
     (40)(50) (60)
      /
     7 (70)
```

Subtree sizes: $s(1)=7$, $s(2)=4$, $s(3)=2$, $s(4)=2$, $s(5)=1$, $s(6)=1$, $s(7)=1$. Heavy children: at $1$, children $2$ (size $4$) and $3$ (size $2$), so heavy child is $2$; at $2$, children $4$ (size $2$) and $5$ (size $1$), heavy is $4$; at $4$, only child $7$, heavy is $7$; at $3$, only child $6$, heavy is $6$. The heavy edges $1\!-\!2$, $2\!-\!4$, $4\!-\!7$ form one chain with head $1$; $3\!-\!6$ is a chain with head $3$; and $5$ is a chain of length one. Three chains for seven nodes.

Heavy-first DFS from $1$: I descend the heavy chain $1,2,4,7$ first, handing out `pos` $0,1,2,3$; then unwind to take light branches — node $5$ (light child of $2$) gets `pos` $4$; then node $3$ (light child of $1$) gets `pos` $5$ and its heavy child $6$ gets `pos` $6$. So `pos` $=\{1\!:\!0,\,2\!:\!1,\,4\!:\!2,\,7\!:\!3,\,5\!:\!4,\,3\!:\!5,\,6\!:\!6\}$, and `head` $=\{1\!:\!1,2\!:\!1,4\!:\!1,7\!:\!1,\,5\!:\!5,\,3\!:\!3,6\!:\!3\}$. Reading the values into `pos` order gives the array $[10,20,40,70,50,30,60]$ — and the chain $1,2,4,7$ does sit in the contiguous block of positions $0,1,2,3$, exactly the contiguity the heavy-first DFS was supposed to produce. (I ran the build code on this tree and it printed precisely these `pos`, `head`, and array values, so the second-pass stack ordering is doing what I described.)

Now query the path between $7$ and $6$. By hand the path is $7\to4\to2\to1\to3\to6$, and its sum is $70+40+20+10+30+60 = 230$. Let me run the climb and see if it lands there without my ever computing the LCA. Start $u=7,\ v=6$. Heads: `head[7]`$=1$ (depth $0$), `head[6]`$=3$ (depth $1$). The deeper head is `head[6]`, so I lift $v=6$: aggregate `[pos[head[6]], pos[6]]` $=$ `[pos[3], pos[6]]` $=$ `[5,6]`, which is nodes $3$ and $6$, values $30+60 = 90$. Jump $v$ to `parent[head[6]]` $=$ `parent[3]` $=1$. Now $u=7,\ v=1$, and `head[7]` $=$ `head[1]` $=1$ — heads coincide, loop ends after exactly one iteration. The shallower pointer is $v=1$: that is the LCA, and I never asked for it directly; it just fell out as the surviving pointer. Final range `[pos[1], pos[7]]` $=$ `[0,3]`, nodes $1,2,4,7$, values $10+20+40+70 = 140$. Total $90 + 140 = 230$. That matches the by-hand path sum, the meeting node $1$ was counted once (only in the final range, not in the chain slice for $6$), and no node was skipped. I also re-ran the code's `path_query(7,6)` and it returned $230$, and checked all $49$ ordered pairs $(u,v)$ on this tree against a brute-force path walk — every one agreed. So the off-by-one I was worried about isn't there.

One more thing the trace settles: the climb never inspected the values, only positions and depths — at each step it asked the segment tree to combine a range, whatever "combine" means. So swapping the tree's combine to `max` (and identity to $-\infty$) leaves the climb untouched and returns the path maximum; the path-walking logic is aggregate-agnostic, and only the segment tree changes.

`update(u, val)` is now trivial: set the value at node $u$, which lives at `pos[u]` in the array, so it's one segment-tree point update, $O(\log n)$. Since the climb reads current segment-tree values, updates and queries interleave correctly online — no precomputation of answers that an update could invalidate.

Costs. Building: I need subtree sizes, parents, depths, and the heavy child of every node (first pass), then the heavy-first DFS that assigns `head` and `pos` (second pass), then the segment tree. Each pass is $O(n)$, the segment tree build is $O(n)$, so building is $O(n)$. `update` is one point update, $O(\log n)$. `path_query` runs the climb $O(\log n)$ times (that's the chain bound), and each iteration does one segment-tree range query at $O(\log n)$, so $O(\log^2 n)$ per query. Over $q$ queries that's $O(n + q\log^2 n)$ — at $n, q \sim 10^5$ this is a few times $10^6 \cdot \log$, comfortably fast. The $\log^2$ is the price of letting the chains be vertical; the alternative single-$\log$ tricks (prefix aggregates per chain) exist but need the aggregate to be invertible/static, and I want plain online point-update with sum *or* max, so $O(\log^2 n)$ per query it is.

One implementation hazard at $n = 10^5$: a recursive DFS for the size/heavy pass and for the decompose pass would recurse to depth $n$ on a bamboo and blow the interpreter stack. So I'll do both passes with explicit stacks. The first pass: push the root, pop-and-record nodes setting parent/depth on the way down to get a visit order, then walk that order in reverse (children before parents) to accumulate subtree sizes and pick the heavy child as the running-largest child. The decompose pass: an explicit stack carrying `(node, chain_head)`; pop a node, assign its `head` and the next `pos`; then to make the heavy child come out *next* off the stack (so its chain stays contiguous), push the light children first and the heavy child last, since the stack is LIFO. Then build the array in `pos` order and hand it to the segment tree.

```python
import sys


def read_tree(data):
    """Parse n, the n node values, and n-1 edges (1-based in input) into a
    0-based adjacency list. Returns (n, values, adj)."""
    it = iter(data)
    n = int(next(it))
    values = [int(next(it)) for _ in range(n)]
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    return n, values, adj


class SegmentTree:
    """Array-backed segment tree: point update, range query under an
    associative combine. Defaults to sum (swap _combine/_identity for max)."""

    def __init__(self, base):
        self.n = len(base)
        self.t = [0] * (2 * self.n)
        for i in range(self.n):
            self.t[self.n + i] = base[i]
        for i in range(self.n - 1, 0, -1):
            self.t[i] = self._combine(self.t[2 * i], self.t[2 * i + 1])

    @staticmethod
    def _combine(a, b):
        return a + b

    _identity = 0

    def update(self, i, val):
        i += self.n
        self.t[i] = val
        i >>= 1
        while i:
            self.t[i] = self._combine(self.t[2 * i], self.t[2 * i + 1])
            i >>= 1

    def query(self, l, r):
        """Aggregate over the index range [l, r] inclusive."""
        res = self._identity
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                res = self._combine(res, self.t[l])
                l += 1
            if r & 1:
                r -= 1
                res = self._combine(res, self.t[r])
            l >>= 1
            r >>= 1
        return res


class TreePaths:
    def __init__(self, n, values, adj, root=0):
        self.n = n
        self.values = values
        self.adj = adj
        self.parent = [-1] * n
        self.depth = [0] * n
        self.size = [1] * n
        self.build(root)

    def build(self, root):
        n, adj = self.n, self.adj
        self.heavy = [-1] * n
        self.head = [0] * n
        self.pos = [0] * n
        parent, depth, size, heavy = self.parent, self.depth, self.size, self.heavy

        # First pass (iterative): parent, depth, subtree size, heavy child.
        order = []
        stack = [root]
        parent[root] = -1
        depth[root] = 0
        visited = [False] * n
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            order.append(v)
            for c in adj[v]:
                if c != parent[v]:
                    parent[c] = v
                    depth[c] = depth[v] + 1
                    stack.append(c)
        for v in reversed(order):          # children before parents
            best = 0
            for c in adj[v]:
                if c != parent[v]:
                    size[v] += size[c]
                    if size[c] > best:
                        best = size[c]
                        heavy[v] = c

        # Second pass (iterative): heavy-child-first DFS assigns chain heads and
        # contiguous positions. Stack carries (vertex, chain head).
        head, pos = self.head, self.pos
        cur = 0
        stack = [(root, root)]
        while stack:
            v, h = stack.pop()
            head[v] = h
            pos[v] = cur
            cur += 1
            # Push light children first so the heavy child is processed next
            # (LIFO), keeping each heavy chain contiguous in pos.
            for c in adj[v]:
                if c != parent[v] and c != heavy[v]:
                    stack.append((c, c))
            if heavy[v] != -1:
                stack.append((heavy[v], h))

        base = [0] * n
        for v in range(n):
            base[pos[v]] = self.values[v]
        self.seg = SegmentTree(base)

    def update(self, u, val):
        self.seg.update(self.pos[u], val)

    def path_query(self, u, v):
        res = SegmentTree._identity
        head, pos, parent, depth = self.head, self.pos, self.parent, self.depth
        while head[u] != head[v]:
            if depth[head[u]] < depth[head[v]]:
                u, v = v, u
            res = SegmentTree._combine(res, self.seg.query(pos[head[u]], pos[u]))
            u = parent[head[u]]
        if depth[u] > depth[v]:
            u, v = v, u
        res = SegmentTree._combine(res, self.seg.query(pos[u], pos[v]))
        return res


def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n, values, adj = read_tree(data)
    tp = TreePaths(n, values, adj)
    # (driver would read and dispatch queries here)


if __name__ == "__main__":
    main()
```

The whole chain of reasoning: a path can sweep $\Theta(n)$ nodes, so re-summing it per query is $O(nq)$ and hopeless; a segment tree would give $O(\log n)$ per query if the path's nodes lived in a few contiguous array ranges, but plain DFS order makes subtrees contiguous and vertical ancestor-chains scattered; so I cut the tree into chains by always continuing into the largest-subtree (heavy) child, which forces any root-to-node path to cross only $O(\log n)$ light edges because each light step at least halves the subtree size; a heavy-child-first DFS lays every chain into a contiguous block of one flat array, so one segment tree handles all updates and range queries; and a path query climbs chain by chain, always lifting the deeper-headed pointer, which both aggregates each chain slice in one range query and lands the two pointers on the LCA — $O(\log n)$ chains times $O(\log n)$ per range query, $O(\log^2 n)$ per query, with $O(n)$ build and $O(\log n)$ updates.
