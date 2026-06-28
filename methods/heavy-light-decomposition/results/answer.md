# Heavy-light decomposition for path queries on a tree

## Problem

A tree on $n$ nodes, each holding a value. Support online: set the value at a node;
and report the sum *or* the maximum of the values on the unique simple path between
two nodes $u$ and $v$. With $n, q$ up to $\sim 10^5$ a per-query walk of the path is
$O(n)$ each, $O(nq)$ overall, too slow. The judged form (ZJOI2008 "Tree Statistics")
fixes the I/O contract used below: read $n$, the $n-1$ tree edges, the $n$ node
weights, then $q$ operations `CHANGE u t` / `QMAX u v` / `QSUM u v`, answering each
online before the next is read.

## Key idea

A segment tree answers point-update + range-aggregate in $O(\log n)$ — *if* the
path's nodes form a few contiguous array ranges. Plain DFS order makes subtrees
contiguous but scatters vertical ancestor-chains, which is exactly what a path is
made of. The fix is to choose the linear order.

**Heavy / light edges.** Root the tree. The **heavy child** of a node is the
child with the largest subtree (ties arbitrary); the edge to it is the **heavy
edge**, all other downward edges are **light**. Maximal runs of heavy edges are
**heavy chains**, and they partition the nodes (a node with no heavy child is a
length-one chain).

**The $O(\log n)$ chain bound.** Descending a light edge from $v$ into a
non-largest child $c$ at least halves the subtree size: the largest child has
subtree $\ge s(c)$, and both are disjoint inside $v$, so $s(v) \ge 2\,s(c)$, i.e.
$s(c) \le s(v)/2$. Subtree size starts at $n$ and bottoms out at $1$, so any
root-to-node path crosses at most $\log_2 n$ light edges — hence touches
$O(\log n)$ chains.

**Contiguous chains in one array.** A DFS that visits the heavy child *first*
gives every heavy chain a contiguous run of positions `pos[]` (chain head
smallest, bottom largest). Store `head[v]` = the top node of $v$'s chain. One
segment tree over the `pos` array then serves all updates and range queries; the
chains partition the $n$ nodes, so the array has exactly $n$ slots.

**Query by chain-climb (LCA for free).** To aggregate the path $u \to v$, while
$u$ and $v$ lie on different chains, lift the pointer whose chain *head is deeper*:
aggregate that chain's slice `[pos[head], pos[node]]`, then jump to
`parent[head]` (one light edge up). After $O(\log n)$ such steps the heads
coincide; the shallower node is the lowest common ancestor, and one final range
over `[pos[shallower], pos[deeper]]` finishes the path. The climb never assumes
which aggregate is used, so swapping the segment tree's combine to `max` answers
path-maximum unchanged.

## Algorithm

1. First DFS (iterative): `parent`, `depth`, subtree `size`, and `heavy[v]` =
   largest-subtree child of $v$.
2. Second DFS (iterative, heavy child first): `head[v]` and `pos[v]`, laying each
   chain into a contiguous block; build the segment trees (sum and max) over the
   values in `pos` order.
3. `CHANGE u t`: point update at `pos[u]` in both trees.
4. `QSUM`/`QMAX u v`: climb chains, always lifting the deeper-headed pointer,
   aggregating each chain slice; finish with the in-chain range through the LCA.

## Code

Single-file C++17, reads from stdin and writes to stdout (no class library): one
`build` plus two segment trees (sum and max over the `pos` array) and the chain-climb
queries, driven by a small `main` that parses the I/O contract above.

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

## Complexity

- **Build:** $O(n)$ — two linear DFS passes (iterative, so $n = 10^5$ bamboos do
  not overflow the stack) plus an $O(n)$ segment-tree build. $O(n)$ memory.
- **`CHANGE` (update):** $O(\log n)$ — one segment-tree point update.
- **`QSUM` / `QMAX` (path query):** $O(\log^2 n)$ — $O(\log n)$ chains crossed,
  each an $O(\log n)$ range query.
- Total over $q$ operations: $O\big(n + q\log^2 n\big)$, ample for
  $n, q \sim 10^5$. Sum and max are carried in two parallel segment trees over the
  same `pos` array; only the combine differs, so the chain-climb is shared. Path
  sums can exceed 32 bits, so weights and accumulators are `long long`, and the
  max-tree's identity is a finite $-\infty$ sentinel (`LLONG_MIN/4`) safe against
  negative weights.
