I am given a tree on $n$ nodes, each carrying a value, and two operations that arrive online and must each be answered before the next is read: `update(u, val)` sets the value at node $u$, and `path_query(u, v)` returns the aggregate — the sum, or just as well the maximum — of the values on the unique simple path between $u$ and $v$, with $n$ and the number of operations both up to about $10^5$. The honest starting point is to walk the path: find the route from $u$ to $v$ and add up the node values. But a single path can sweep through $\Theta(n)$ nodes — on a bamboo, a chain of $n$ nodes, the path from one end to the other is the whole tree — so one query costs $O(n)$ and $q$ of them cost $O(nq)\approx 10^{10}$, which is hopeless. And because updates interleave with queries, I cannot precompute all the answers offline either. The thing that kills me is that I re-sum $\Theta(n)$ nodes from scratch on every query.

For a query over a contiguous *range* of an array I know exactly what to reach for — a segment tree, which does point update and range aggregate each in $O(\log n)$. So the real question is not "how do I sum a path" but "can I lay the tree's nodes into a linear array so that any path becomes a few contiguous slices?" The natural candidate, DFS-visit order `pos[v]`, has one beautiful property: every *subtree* occupies a contiguous block, because DFS enters a subtree, finishes it entirely, and only then leaves. But a path is not a subtree. Split it at the lowest common ancestor $l$ of $u$ and $v$ and it becomes two upward ancestor-chains, $u\to l$ and $v\to l$ — and in plain DFS order the ancestors of a node are scattered all over `pos`, so a vertical chain is in general $\Theta(\text{depth})$ separate positions, not a contiguous run. Plain DFS order gives subtrees for free and does nothing for the vertical chains a path is actually made of. I need a *different* linear order, one tuned so that the chains I query are contiguous.

The method is heavy-light decomposition. Suppose I cut the tree into top-to-bottom chains and lay each chain into a contiguous block of one flat array; then an upward path $u\to l$ crosses some number of chains, and on each crossed chain I do one segment-tree range query, so the per-query cost is (number of chains crossed) $\times\;O(\log n)$. Everything hinges on bounding the number of chains a single root-to-node path can cross — and that number is entirely at my mercy, because *I* choose which downward edges stay within a chain and which start a new one. At each node I let exactly one child's edge continue the current chain and declare the rest chain boundaries; the entire game is to choose the continuing child so that *no* root-to-node descent crosses more than $O(\log n)$ boundaries, no matter the tree's shape. Choosing blindly fails — an adversary nests the wrong choices and a descent zig-zags across $\Theta(n)$ chains — so the choice must be tied to subtree size. The rule is: the continuing child is the one with the **largest subtree**. Call the edge to it the *heavy* edge and the rest *light* edges; maximal runs of heavy edges are *heavy chains*, and since each node has exactly one parent and at most one heavy edge going down, the heavy edges decompose into disjoint top-to-bottom chains that partition all $n$ nodes (a node with no heavy child is a chain of length one).

What this buys is a clean halving bound. When a descent takes a light edge from $v$ into a child $c$ that is *not* the largest, the largest child has subtree at least $s(c)$, and the two children are disjoint pieces of $v$'s subtree, so $s(v)\ge s(c)+s(\text{largest})\ge 2\,s(c)$, giving

$$s(c)\le \frac{s(v)}{2}.$$

Every light step at least halves the subtree size of where I am standing; the size starts at $n$ at the root and cannot drop below $1$, so any root-to-node path takes at most $\log_2 n$ light steps and therefore lies on at most $O(\log n)$ chains. The bound does not care about the tree's shape — that is exactly why the adversary cannot beat the largest-subtree rule, where any blind rule fails. The tie case is harmless: if two children are equal-largest, only one becomes heavy, and the other is then a non-largest child with subtree $\le s(v)/2$, so the halving still holds.

To make each chain a contiguous slice of the array I run a DFS that visits the **heavy child first**, before any light children. Then walking down a node, its heavy child, that child's heavy child, and so on, I assign consecutive `pos` values straight down a whole heavy chain before ever backing up to a light branch — so each chain becomes an interval `[pos[head], pos[bottom]]`, head smallest and bottom largest. I store `head[v]`, the topmost node of $v$'s chain, so from any node I can see which chain it is on and jump to its top. Because the chains partition the $n$ nodes, the array has exactly $n$ slots and a single $O(n)$ segment tree over it serves every update and every range query — one tree, not one per chain. `update(u, val)` is then a single point update at `pos[u]`, $O(\log n)$.

The path query climbs chain by chain, and the lowest common ancestor falls out for free. I keep two pointers $u$ and $v$. While they sit on different chains (`head[u] != head[v]`), I lift the one whose chain *head is deeper*: that node's chain head, being deeper, cannot be an ancestor of the other pointer, so the slice of the path from the chain head down to the pointer — `query(pos[head[u]], pos[u])` — is definitely part of the answer and can be aggregated now, after which I jump that pointer to `parent[head[u]]`, stepping up one light edge onto the next chain. Each iteration moves a pointer up one entire chain, so the loop runs $O(\log n)$ times by the light-edge bound. When the heads finally coincide, both pointers lie on one chain; the shallower of them is the lowest common ancestor $l$, and a final `query` over `[pos[shallower], pos[deeper]]` adds the last stretch through $l$ — I never computed the LCA on the side. Accounting confirms no double-count and no gap: while heads differ I consume `[pos[head[u]], pos[u]]` and then jump strictly above it to the parent of the head, so the consumed and remaining parts abut without overlap; when the heads match the inclusive range covers $l$ exactly once, and the two arms stitched at $l$ count it once. The climb never assumes anything about the aggregate, so swapping the segment tree's `_combine`/`_identity` from sum to max returns the path maximum with no other change. The cost is $O(\log n)$ chains times $O(\log n)$ per range query, i.e. $O(\log^2 n)$ per query, with $O(n)$ build and $O(\log n)$ updates — over $q$ operations, $O(n + q\log^2 n)$, ample at $n,q\sim 10^5$. The extra $\log$ over a single-$\log$ trick is the price of vertical chains under a general associative aggregate; prefix-aggregate-per-chain shortcuts exist but need an invertible or static aggregate, whereas I want plain online point-update with sum *or* max.

One implementation hazard at $n=10^5$: a recursive DFS would recurse to depth $n$ on a bamboo and overflow the stack, so both passes use explicit stacks. The first pass pushes the root, records nodes while setting `parent`/`depth` on the way down, then walks that visit order in reverse — children before parents — to accumulate subtree sizes and pick each node's heavy child as the running-largest child. The decompose pass carries `(vertex, chain_head)` on its stack and, to make the heavy child come off next and keep its chain contiguous, pushes the light children first and the heavy child last, exploiting the stack's LIFO order. Then the array is filled in `pos` order and handed to the segment trees.

The deliverable is a single self-contained C++17 program. It follows the judged form of this task (ZJOI2008 "Tree Statistics"): from stdin it reads $n$, then the $n-1$ tree edges, then the $n$ node weights, then $q$ operations — `CHANGE u t` sets node $u$'s weight, `QMAX u v` and `QSUM u v` report the path maximum and path sum — and prints one line per query to stdout. Two parallel segment trees over the `pos` array carry sum and max so the same chain-climb answers either; weights and accumulators are `long long` because a path sum can exceed 32 bits, and the max-tree's identity is a finite $-\infty$ sentinel so negative weights are handled.

```cpp
// Heavy-light decomposition: online path sum / path max on a weighted tree.
// Reads from stdin: n; then n-1 edges "a b" (1-based); then n node weights;
// then q; then q operations, one per line:
//   CHANGE u t  -> set node u's weight to t
//   QMAX u v    -> print max weight on the path u..v
//   QSUM u v    -> print sum of weights on the path u..v
// Prints one line per QMAX/QSUM query to stdout. (ZJOI2008 "Tree Statistics".)
#include <bits/stdc++.h>
using namespace std;

const long long NEG_INF = LLONG_MIN / 4;

int n;
vector<vector<int>> adj;
vector<long long> wt;           // node weights, 1-based
vector<int> parent, depth_, sz, heavy, head, pos_;

// Two segment trees over the pos[] array: one for sum, one for max.
struct SegSum {
    int m;
    vector<long long> t;
    void init(const vector<long long>& base) {
        m = (int)base.size();
        t.assign(2 * m, 0);
        for (int i = 0; i < m; i++) t[m + i] = base[i];
        for (int i = m - 1; i > 0; i--) t[i] = t[2 * i] + t[2 * i + 1];
    }
    void update(int i, long long val) {
        for (t[i += m] = val, i >>= 1; i; i >>= 1)
            t[i] = t[2 * i] + t[2 * i + 1];
    }
    long long query(int l, int r) { // inclusive [l, r]
        long long res = 0;
        for (l += m, r += m + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res += t[l++];
            if (r & 1) res += t[--r];
        }
        return res;
    }
};

struct SegMax {
    int m;
    vector<long long> t;
    void init(const vector<long long>& base) {
        m = (int)base.size();
        t.assign(2 * m, NEG_INF);
        for (int i = 0; i < m; i++) t[m + i] = base[i];
        for (int i = m - 1; i > 0; i--) t[i] = max(t[2 * i], t[2 * i + 1]);
    }
    void update(int i, long long val) {
        for (t[i += m] = val, i >>= 1; i; i >>= 1)
            t[i] = max(t[2 * i], t[2 * i + 1]);
    }
    long long query(int l, int r) { // inclusive [l, r]
        long long res = NEG_INF;
        for (l += m, r += m + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = max(res, t[l++]);
            if (r & 1) res = max(res, t[--r]);
        }
        return res;
    }
};

SegSum segSum;
SegMax segMax;

// First pass (iterative): parent, depth, subtree size, heavy child.
// Second pass (iterative, heavy child first): head[] and pos[], laying each
// heavy chain into a contiguous block. Iterative to survive depth-n bamboos.
void build(int root) {
    parent.assign(n + 1, 0);
    depth_.assign(n + 1, 0);
    sz.assign(n + 1, 1);
    heavy.assign(n + 1, 0);   // 0 = none (nodes are 1-based)
    head.assign(n + 1, 0);
    pos_.assign(n + 1, 0);

    vector<int> order;
    order.reserve(n);
    vector<char> visited(n + 1, 0);
    vector<int> stk;
    stk.push_back(root);
    parent[root] = 0;
    depth_[root] = 0;
    while (!stk.empty()) {
        int v = stk.back(); stk.pop_back();
        if (visited[v]) continue;
        visited[v] = 1;
        order.push_back(v);
        for (int c : adj[v]) if (c != parent[v]) {
            parent[c] = v;
            depth_[c] = depth_[v] + 1;
            stk.push_back(c);
        }
    }
    for (int i = (int)order.size() - 1; i >= 0; i--) { // children before parents
        int v = order[i];
        int best = 0;
        for (int c : adj[v]) if (c != parent[v]) {
            sz[v] += sz[c];
            if (sz[c] > best) { best = sz[c]; heavy[v] = c; }
        }
    }

    // Heavy-child-first DFS: stack carries (vertex, chain head). Push light
    // children first so the heavy child pops next (LIFO) -> chain stays contiguous.
    int cur = 0;
    vector<pair<int,int>> stk2;
    stk2.emplace_back(root, root);
    while (!stk2.empty()) {
        auto [v, h] = stk2.back(); stk2.pop_back();
        head[v] = h;
        pos_[v] = cur++;
        for (int c : adj[v]) if (c != parent[v] && c != heavy[v])
            stk2.emplace_back(c, c);
        if (heavy[v] != 0) stk2.emplace_back(heavy[v], h);
    }

    vector<long long> base(n, 0);
    for (int v = 1; v <= n; v++) base[pos_[v]] = wt[v];
    segSum.init(base);
    segMax.init(base);
}

void updateNode(int u, long long val) {
    wt[u] = val;
    segSum.update(pos_[u], val);
    segMax.update(pos_[u], val);
}

// Climb chain by chain, always lifting the deeper-headed pointer. The LCA
// falls out as the surviving pointer; the climb is aggregate-agnostic.
long long pathSum(int u, int v) {
    long long res = 0;
    while (head[u] != head[v]) {
        if (depth_[head[u]] < depth_[head[v]]) swap(u, v);
        res += segSum.query(pos_[head[u]], pos_[u]);
        u = parent[head[u]];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    res += segSum.query(pos_[u], pos_[v]);
    return res;
}

long long pathMax(int u, int v) {
    long long res = NEG_INF;
    while (head[u] != head[v]) {
        if (depth_[head[u]] < depth_[head[v]]) swap(u, v);
        res = max(res, segMax.query(pos_[head[u]], pos_[u]));
        u = parent[head[u]];
    }
    if (depth_[u] > depth_[v]) swap(u, v);
    res = max(res, segMax.query(pos_[u], pos_[v]));
    return res;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n)) return 0;
    adj.assign(n + 1, {});
    wt.assign(n + 1, 0);
    for (int i = 0; i < n - 1; i++) {
        int a, b; cin >> a >> b;
        adj[a].push_back(b);
        adj[b].push_back(a);
    }
    for (int i = 1; i <= n; i++) cin >> wt[i];
    build(1);
    int q; cin >> q;
    string op; int u, v;
    while (q--) {
        cin >> op >> u >> v;
        if (op == "CHANGE") updateNode(u, v);
        else if (op == "QMAX") cout << pathMax(u, v) << '\n';
        else if (op == "QSUM") cout << pathSum(u, v) << '\n';
    }
    return 0;
}
```
