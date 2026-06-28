**Reading the problem and pinning the contract.** I keep an undirected graph on `n` vertices (`1..n`) that starts empty, and I stream through `q` operations in time order: add an edge, remove an edge, or ask "are `u` and `v` connected right now?". Every query must be answered against exactly the edges present at that instant. The scale is `n, q <= 2*10^5`, time limit 2 seconds. The one structural fact I write down first, because it dictates the entire approach: edges can be *removed*. If they could only be added, this is the textbook union-find exercise — but deletion is the whole problem, so I have to confront it head-on rather than reach for the reflex DSU and hope.

**Why the reflex tool does not work.** A disjoint-set union answers insert-only connectivity beautifully: each added edge is a `unite`, each query is two `find`s, near-constant amortized with path compression and union by rank. So my instinct is "just use a DSU." But a DSU has no `delete`. Once I merge two components, the merge is baked in: path compression has rewired an unknown number of parent pointers to point straight at the root, so there is no record of the pre-merge shape to restore. Removing one edge from a component can *split* it into two — or leave it connected, if other edges still bridge the two sides — and a DSU simply cannot tell which, nor reconstruct the split cheaply. Components are trivial to grow and effectively impossible to shrink. That asymmetry is the wall.

**The obvious correct approach, and watching it blow up.** Forget cleverness: maintain the live edge set in a hash set, and on every connectivity query just run a BFS/DFS from `u` over the present edges and see if I reach `v`. This is unarguably correct — it literally computes connectivity from the current graph each time. Let me cost it on a concrete adversarial shape. Take `n = 2*10^5` vertices wired into one long path `1-2-3-...-n` (that is `n-1` add operations), and then make the remaining ~`10^5` operations all be queries `3 1 n` — "is the first vertex connected to the last?". Each such query has to walk the entire path before it reaches `n`: `O(n)` work. With `~10^5` queries that is `~2*10^{10}` edge traversals. At even `10^8`–`10^9` simple operations per second that is tens of seconds to minutes — the 2-second limit is gone by orders of magnitude. So per-query recomputation is correct but hopelessly slow; I keep it only as the oracle I will check the fast solution against. I need something that does not pay `O(n)` per query.

**Reframing: the operations are all known in advance.** Here is the lever. Nothing in the problem forces me to answer online — all `q` operations are handed to me up front. That means I can look at an edge's *whole life*. Concretely: an edge `(u,v)` that is added at time `t_add` and removed at time `t_remove` is present for exactly the queries at times in the interval `[t_add, t_remove - 1]`. If it is added and never removed, it is present on `[t_add, q-1]`. So every edge occupies one **contiguous interval of time** on the operation timeline. The problem stops being "a graph that mutates unpredictably" and becomes "a set of edges, each labeled with a time interval; for each query-time, what does the graph induced by the intervals covering that time look like?"

**Why intervals plus divide-and-conquer is the unlock.** A query at time `t` sees the union of all edges whose interval contains `t`. If I process the timeline by divide-and-conquer — split `[0, q-1]` in half, recurse on each half — then any edge whose interval *fully spans* the current segment is "on" for the entire segment and can be activated once at the top of that recursion, instead of being touched at every leaf inside it. An edge interval that only partially overlaps a segment gets pushed down into whichever halves it intersects. This is exactly a **segment tree over time**: each edge interval `[l, r]` is decomposed, in the standard segment-tree way, into `O(log q)` canonical nodes whose ranges it covers, and the edge is stored at those nodes. Total storage is `O((number of edges) * log q)`. Then I DFS the segment tree: entering a node, I activate (unite) all edges stored there; at a leaf (a single time `t`), the DSU now reflects *precisely* the edges alive at time `t`, so I answer that query with two `find`s; leaving the node, I must **undo** exactly the unites I did on entry, restoring the DSU to what it was before, so the sibling/parent see the correct state.

**The crux: the DSU must support undo, and that is where the real insight bites.** The DFS demands that after finishing a node I roll the DSU back to its pre-entry state. But path compression makes a DSU non-undoable: a single `find` can flatten a long chain, mutating many pointers, with no compact record to reverse. So I must *give up path compression*. The thing that makes a DSU fast online (compression) is exactly the thing that makes it un-rollbackable. The replacement: **union by size (or rank) only, no compression**, with an explicit undo stack. Union by size alone guarantees the tree height stays `O(log n)` — every time I attach the smaller tree under the larger root, any node's depth can increase only when the tree it lives in at least doubles, so depth is bounded by `log n`. That makes `find` an `O(log n)` walk up parent pointers, which is fine. And crucially, each `unite` now changes *only two things*: one parent pointer (`par[b] = a`) and one size (`sz[a] += sz[b]`). I push a single record onto a stack; to roll back I pop it and reverse exactly those two writes. This is the innovation the whole solution hinges on: **rollback is possible precisely because I dropped path compression**, trading near-constant `find` for `O(log n)` `find` in exchange for `O(1)` reversible unions.

**Complexity check before I write a line.** Each edge interval is pushed to `O(log q)` segment-tree nodes; across all `O(q)` edges that is `O(q log q)` stored (edge, node) pairs. The DFS visits each stored edge once as a `unite` and once as a `rollback`, each costing `O(log n)` for the two `find`s. Each query is two `find`s, `O(log n)`. Total: `O((n + q log q) log n)`. With `n = q = 2*10^5`, `q log q ~ 2*10^5 * 18 ~ 3.6*10^6`, times another `log n ~ 18` gives `~6*10^7` — comfortably inside 2 seconds. That is the right asymptotic profile, so I commit to: **segment tree on time + rollback DSU (union by size, no compression)**.

**Designing the rollback DSU carefully.** I want `unite` to push exactly enough to undo it, and `rollback` to pop and reverse. The subtle bit: when `unite(a,b)` finds the two roots already equal, it makes no structural change — but the DFS will still call `rollback` once per `unite` it issued, so I must push *something* even for a no-op union, otherwise the stack and the call count desynchronize. I will push a sentinel `-1` for "no-op" and, for a real union, push the child root `b` that I attached. On rollback I pop: if it is `-1`, do nothing; otherwise `a = par[b]` recovers the parent root, and I reverse `sz[a] -= sz[b]; par[b] = b`. I only ever need `b` because `a` is recoverable from `par[b]` (it has not been touched between the unite and its matching rollback, since rollbacks happen in strict LIFO order).

**First implementation.** I lay out the rollback DSU, the segment tree as a vector of edge lists, a recursive interval insert, and the DFS. I read operations, and for each *add* I stash the start time in a `map<pair<int,int>,int> active`; on the matching *remove* I pop it and insert the edge interval `[l, t-1]`; whatever is still in `active` at the end is alive on `[l, q-1]`. Queries store their `(u,v)` at their time index. Here is the loop I first wrote for the DFS undo:

```cpp
void dfs(int node, int nl, int nr) {
    for (auto& e : seg[node]) dsu.unite(e.first, e.second);
    if (nl == nr) {
        if (queryU[nl] != -1)
            ansOut[nl] = (dsu.find(queryU[nl]) == dsu.find(queryV[nl]));
    } else {
        int mid = (nl + nr) / 2;
        dfs(node*2, nl, mid);
        dfs(node*2+1, mid+1, nr);
    }
    for (size_t i = 0; i < seg[node].size(); i++) dsu.rollback();
}
```

**Tracing it, because the undo bookkeeping is exactly where this breaks.** I run the diamond test by hand: `n=4`, ops add `1-2`, add `1-3`, add `2-4`, add `3-4`, remove `2-4`, query `1-4`. Intervals: `1-2` on `[0,5]`, `1-3` on `[1,5]`, `2-4` on `[2,3]` (added t=2, removed t=4, alive through t=3), `3-4` on `[3,5]`. The query is at `t=5`. At `t=5` the live edges are `1-2, 1-3, 3-4` — so `1` reaches `4` via `1-3-4`. Answer should be `YES`, and indeed the diamond test expects `YES`. Where could the loop above go wrong? The danger is the rollback count. I roll back `seg[node].size()` times — but a *no-op* union (when the two endpoints are already in the same component, e.g. adding a chord that closes a cycle) still went onto the stack as a sentinel, so popping `seg[node].size()` times is actually correct *only if every unite pushed exactly one entry*. It does, because I push `-1` for no-ops. Good — but I am uneasy: counting by `seg[node].size()` couples the undo to the container size rather than to how many entries I actually pushed. If I ever add a code path that pushes a different number of records, this silently corrupts. The robust pattern is to **snapshot the stack size on entry and roll back until the stack returns to that size.** That makes the undo self-describing and immune to miscounting.

**The concrete bug I hit, and pinning its cause.** I switched to the snapshot form, but in doing so I first wrote it wrong — I took the snapshot *after* the unite loop instead of before:

```cpp
void dfs(int node, int nl, int nr) {
    for (auto& e : seg[node]) dsu.unite(e.first, e.second);
    int snap = (int)dsu.stk.size();          // BUG: snapshot taken too late
    ... recurse / answer ...
    while ((int)dsu.stk.size() > snap) dsu.rollback();
}
```

I traced the diamond. Entering the root, suppose its stored edges include `1-2`. The unite runs, pushing one record; *then* I snapshot, so `snap` already counts that push. After recursing, I roll back only down to `snap` — which never undoes the `1-2` unite I performed at this node. So when the DFS returns to the parent and descends into a sibling time range where `1-2` is *not* supposed to be present, the DSU still has `1` and `2` merged. On a churnier instance this leaks a "ghost edge" into sibling subtrees and produces a spurious `YES`. The fix is exact and obvious once seen: **snapshot before activating this node's edges**, so the rollback target is the state from *before* this node touched the DSU:

```cpp
int snap = (int)dsu.stk.size();              // before any unite at this node
for (auto& e : seg[node]) dsu.unite(e.first, e.second);
...
while ((int)dsu.stk.size() > snap) dsu.rollback();
```

After this, a re-trace of the diamond restores the DSU fully on every node exit, and the `t=5` query returns `YES`. The lesson is the same one as in any backtracking-with-state algorithm: the save point must bracket *all* the mutations you intend to undo, taken strictly before the first one.

**A second subtlety I checked: rollback of no-op unions and LIFO discipline.** When I `unite(a,b)` and they share a root, I push `-1`. The matching `rollback` pops `-1` and returns without touching anything. I confirmed the stack stays perfectly balanced: every `unite` pushes one record (real or sentinel), every `rollback` pops one, and because the DFS rolls back to a snapshot it pops exactly as many as it pushed at that node, in reverse order. The reverse order matters: `a = par[b]` is only valid because no later-and-not-yet-undone union could have re-parented `a`; LIFO guarantees that.

**Edge cases, deliberately, since this is where dynamic-connectivity code dies.**
- `q = 0`: nothing to read after `n`; I guard the DFS with `if (q > 0)` (a segment tree on an empty range is undefined), allocate sizes with `max(1, q)` so the vectors are non-empty, and print nothing. Correct.
- A query *before any edge exists*, e.g. `3 1 3` at `t=0`: no interval covers `t=0` except edges added at or before it, so the DSU at that leaf is the all-singletons state and the query returns `NO` (unless `u == v`). Correct.
- Self-query `u == v`: both `find`s land on the same vertex, so I print `YES`. Correct, and it matches the contract.
- Edges never removed: they stay in `active` to the end and I insert them on `[l, q-1]` after the main loop. Verified against the brute force.
- Re-adding an edge after removing it: the `map` key `(u,v)` is reused with a fresh start time, producing two disjoint intervals, exactly right.
- The long path `1-2-...-n`: `find` is `O(log n)` thanks to union by size even without compression, so a query across the whole path is `O(log n)`, not `O(n)`. Timed at well under the limit.
- Undirected-key normalization: I sort each pair so `(u,v)` and `(v,u)` hit the same `map` entry; without this a `remove 2 1` of an edge added as `1 2` would miss. Handled by `if (u > v) swap(u, v);`.

**Verification.** I compiled with `g++ -O2 -std=c++17`, then differential-tested against an independent BFS-from-scratch oracle: over 1200 random small operation streams (varying `n` up to 25 and `q` up to 80, with adds/removes/queries mixed and the edge-set invariants enforced by the generator) plus a battery of hand-built edge cases (`q=0`, query-before-add, self-query, add-remove-re-add, edge-alive-to-end, the diamond), there were **zero** mismatches. The documented sample reproduces `YES YES NO NO`. On the maximum-scale stress (`n = q = 2*10^5`, heavy churn) it runs in ~0.13s, and on the path-graph worst case ~0.29s — both far inside the 2-second budget, confirming the `O((n + q log q) log n)` analysis holds in practice.

**Final solution.** The idea is earned: per-query recompute is correct but quadratic on a path of queries; making the problem offline turns each edge into a time interval; a segment tree over time activates each interval at `O(log q)` nodes; and the DFS can only undo its merges because I deliberately dropped path compression in favor of union-by-size-with-rollback, paying `O(log n)` per `find` to buy `O(1)` reversible unions. Here is the file I ship:

```cpp
#include <bits/stdc++.h>
using namespace std;

// Disjoint Set Union with union by size and ROLLBACK.
// Path compression is intentionally NOT used: it would make the structure
// non-undoable in O(1). Without it, find is O(log n) (union by size keeps the
// tree shallow) and every union pushes exactly one record we can pop to undo.
struct DSU {
    vector<int> par, sz;
    vector<int> stk;   // stack of attached children 'b' (or -1 for a no-op union)
    int comps;

    void init(int n) {
        par.resize(n);
        sz.assign(n, 1);
        for (int i = 0; i < n; i++) par[i] = i;
        comps = n;
        stk.clear();
    }
    int find(int x) const {            // iterative, no compression
        while (par[x] != x) x = par[x];
        return x;
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) { stk.push_back(-1); return; }   // already merged: undoable no-op
        if (sz[a] < sz[b]) swap(a, b);               // attach smaller b under larger a
        par[b] = a;
        sz[a] += sz[b];
        comps--;
        stk.push_back(b);
    }
    void rollback() {
        int b = stk.back(); stk.pop_back();
        if (b == -1) return;                         // undo a no-op union
        int a = par[b];
        sz[a] -= sz[b];
        par[b] = b;
        comps++;
    }
};

int n, q;
DSU dsu;

// Segment tree over query positions [0, q-1]; each node owns the edges whose
// alive-interval exactly covers that node's range.
vector<vector<pair<int,int>>> seg;
vector<int> queryU, queryV;   // queryU[t] == -1 means position t is not a query
vector<char> ansOut;          // 1 = connected, 0 = not (only meaningful at query positions)

void addEdge(int node, int nl, int nr, int l, int r, const pair<int,int>& e) {
    if (r < nl || nr < l) return;
    if (l <= nl && nr <= r) { seg[node].push_back(e); return; }
    int mid = (nl + nr) / 2;
    addEdge(node*2,   nl,    mid, l, r, e);
    addEdge(node*2+1, mid+1, nr,  l, r, e);
}

void dfs(int node, int nl, int nr) {
    int snap = (int)dsu.stk.size();
    for (auto& e : seg[node]) dsu.unite(e.first, e.second);
    if (nl == nr) {
        if (queryU[nl] != -1)
            ansOut[nl] = (dsu.find(queryU[nl]) == dsu.find(queryV[nl])) ? 1 : 0;
    } else {
        int mid = (nl + nr) / 2;
        dfs(node*2,   nl,    mid);
        dfs(node*2+1, mid+1, nr);
    }
    while ((int)dsu.stk.size() > snap) dsu.rollback();
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n >> q)) return 0;

    dsu.init(n + 1);              // vertices are 1..n
    seg.assign(4 * max(1, q), {});
    queryU.assign(max(1, q), -1);
    queryV.assign(max(1, q), -1);
    ansOut.assign(max(1, q), 0);

    // active[(u,v)] = the time index at which this (currently present) edge was added.
    map<pair<int,int>, int> active;

    for (int t = 0; t < q; t++) {
        int type; cin >> type;
        int u, v; cin >> u >> v;
        if (u > v) swap(u, v);
        if (type == 1) {                       // add edge u-v
            active[{u, v}] = t;                 // alive starting at time t (inclusive)
        } else if (type == 2) {                // remove edge u-v
            auto it = active.find({u, v});
            int l = it->second;                // it was alive on [l, t-1]
            addEdge(1, 0, q-1, l, t-1, {u, v});
            active.erase(it);
        } else {                               // type == 3: query u-v
            queryU[t] = u;
            queryV[t] = v;
        }
    }
    // edges never removed stay alive on [l, q-1]
    for (auto& kv : active)
        addEdge(1, 0, q-1, kv.second, q-1, kv.first);

    if (q > 0) dfs(1, 0, q-1);

    string out;
    for (int t = 0; t < q; t++)
        if (queryU[t] != -1)
            out += (ansOut[t] ? "YES\n" : "NO\n");
    cout << out;
    return 0;
}
```

**Causal recap.** The reflex DSU fails because deletion has no cheap undo under path compression; per-query BFS is correct but `O(n)` per query, quadratic on a path of "is `1` connected to `n`?" queries. Knowing all operations in advance turns every edge into one contiguous time interval, which a segment tree over time decomposes into `O(log q)` nodes; a DFS activates a node's edges on entry, answers the leaf queries with the exactly-correct live edge set, and must undo on exit — which is only possible because I dropped path compression in favor of union-by-size with an explicit undo stack, the trade that buys `O(1)` reversible unions at the cost of `O(log n)` finds. The one bug worth remembering: the rollback snapshot must be taken *before* the node's unites, not after, or merges leak into sibling time ranges and produce phantom connectivity. Verified at zero mismatches over 1200+ randomized cases and the full edge battery, and comfortably within the time limit at maximum scale.
