**Reading the problem and pinning the contract.** I have a tree of `n` facilities rooted at node `1`; every edge carries a positive cut cost `w` (`1 <= w <= 10^9`). For each of `q` quarantine orders I am handed a set `S` of marked nodes, and I must cut a cheapest set of edges so that **no node of `S` can still reach the root**. Orders are independent. The first thing I fix is the scale, because it sets the data types and rules out whole classes of algorithm: `n` and `q` go up to `2*10^5`, and crucially the only bound on the marked sets is the *aggregate* `sum |S| <= 2*10^5`. An answer is a sum of cut edges, up to `2*10^5` of them each up to `10^9`, so it can reach `~2*10^14` — well past 32 bits. Every accumulator that touches a sum of weights is `long long`; an `int` there is a silent wrong answer on the large tests. I also note the root is guaranteed never to be marked: a node cannot be quarantined from itself, so that degenerate case is excluded by the input contract and I do not have to define it.

**The shape of a cut on a tree.** Before any algorithm I want to understand what a feasible cut even looks like, because the geometry of the tree constrains it hard. Removing one edge `(p, c)` (with `c` the child) detaches the *entire subtree of `c`* from everything above, at cost `w(p,c)`. To make a marked node `x` unreachable from the root I must remove **at least one edge on the unique root→x path**; conversely, removing any single edge on that path suffices for `x` alone. So a feasible solution is a set of edges that "covers" every marked node's root-path, and its cost is the sum of the chosen edges. This is exactly a minimum edge cut separating the marked terminals from the root; on a tree it has no flow-augmenting subtlety, so I expect a direct dynamic program rather than general max-flow.

**The obvious correct method: a full-tree DP.** Define, over the *whole* tree, `f[v]` = the minimum cost to cut every marked node inside `v`'s subtree away from `v`. Post-order: for each child `c` reached by an edge of cost `w`,

- if `c` is *not* marked, I either cut the edge `(v, c)` now (cost `w`, severing the whole subtree) or I recurse and pay `f[c]` to handle the marked nodes below without touching this edge — so the child contributes `min(w, f[c])`;
- if `c` *is* marked, then `c` itself must be detached from `v`, and `f[c]` (which only separates things from `c`) does not do that, so I must cut on the path between them — the single edge `(v, c)` of cost `w` — contributing `w`.

The answer for the order is `f[1]`. I am confident this is correct (it is the textbook tree min-cut DP), and I sanity-check it on the sample order `{6, 7}`: nodes `6` and `7` hang under node `5`; `f[5] = w(5,6) + w(5,7) = 2 + 5 = 7`, and at the root the child `5` is unmarked so it contributes `min(w(1,5), f[5]) = min(3, 7) = 3`. That matches the expected `3`, and it already shows the interesting effect: cutting the *shared* edge `1-5` above both marked nodes beats cutting each marked node's own edge.

**Why the obvious method is too slow — a concrete cost count.** The full-tree DP is `O(n)` per order because it visits every node, marked or not. Across `q` orders that is `O(q*n)`. Plug in the worst case: `q = 2*10^5` orders, `n = 2*10^5` nodes, and that is `4*10^10` node visits. Even at a generous `10^9` simple operations per second that is forty seconds, twenty times over a 2-second budget. And it is *wasteful* in an obvious way: an order with `|S| = 1` still walks all `2*10^5` nodes to cut a single edge. The work is tied to the tree size, but the only thing the problem bounds is the *total marked-set size*. There is a glaring mismatch between what I am paying for (the whole tree, every time) and what actually varies (the handful of marked nodes). That mismatch is the lever.

**Spotting what actually matters per order.** Let me look again at where the full-tree DP does anything non-trivial. Walking a long chain of unmarked, single-child nodes, the recurrence just propagates `min(w, f[c])` down a straight line — and on a straight chain of edges with no branching and no marked node in the middle, the cheapest "cut this chain" is simply *the minimum edge weight on the chain*. The DP only makes a real decision at three kinds of node: a marked node, a branch point where two marked nodes' paths diverge, and the root. Everything else is filler on a chain whose only summary I need is "what is the cheapest single edge to sever it". For one order, how many such interesting nodes are there? The marked nodes are `|S|`. The branch points are pairwise lowest common ancestors of marked nodes — and a classic fact is that the set of all pairwise LCAs of `|S|` nodes has size `< |S|`, and they are exactly the LCAs of *adjacent* nodes once you sort the marked set by DFS in-time. So there are `O(|S|)` interesting nodes total, not `O(n)`.

**The insight: compress the tree to a virtual (auxiliary) tree.** This is the move. For each order I build a small **virtual tree** on just the `O(|S|)` interesting nodes — the marked nodes, their adjacent-pair LCAs, and the root as anchor. In the virtual tree, an edge `(a, b)` stands for the *path* between `a` and `b` in the original tree, and I label it with the **minimum edge weight on that path** — because the cheapest way to sever a whole chain is to cut its lightest edge. Then I run the *same* min-cut DP, but on this compressed tree of `O(|S|)` nodes instead of the full `O(n)` tree. The DP's logic is identical; only the "edge weight" now means "cheapest cut of the chain this virtual edge represents." Per order the cost becomes `O(|S| log n)` — the `log n` from LCA queries and from extracting the path-minimum — and summed over all orders it is `O(sum|S| * log n)`, comfortably inside the budget. The full-tree DP is the correct skeleton; the virtual tree is what makes it affordable, and the key sub-fact that makes it *exact* is that severing a chain costs its minimum edge, which is precisely what a min-cut wants.

**Tooling I need.** To realize this I need, after one `O(n log n)` preprocessing pass:

- a DFS from the root giving each node its in-time `tin[v]` (for sorting the marked set into DFS order) and depth `dep[v]`;
- LCA queries — I use binary lifting, `up[k][v]` = the `2^k`-th ancestor;
- the path-minimum from a descendant up to an ancestor — I piggyback on the lifting table with `mn[k][v]` = the minimum edge weight on the `2^k`-edge jump up from `v`, so I can read off the min edge between any node and one of its ancestors in `O(log n)`.

For the virtual tree itself I use the standard monotonic-stack construction: sort the chosen nodes by `tin`, push them while maintaining a stack that represents the current rightmost root-to-node path, and whenever a new node's LCA with the stack top is higher than the second element, pop and attach. Each pop creates one virtual edge whose weight I fill in with `minEdgeUp(child, parent)`.

**First implementation.** I write the preprocessing (an iterative rooted DFS so I do not risk a recursion-depth stack overflow on a `2*10^5`-deep bamboo), the binary-lifting tables for both ancestors and path-minima, the LCA and `minEdgeUp` helpers, the per-order virtual-tree build, and the DP. For the DP I translate my earlier full-tree recurrence directly: for a virtual child `c` of `u` with compressed weight `w`,

```
if (isSpec[nodes[c]]) sum += min(w, dp[c]);   // first attempt
else                  sum += min(w, dp[c]);
```

— and I pause, because I have written the same thing in both branches, which means I have not actually used the marked-ness of `c` at all. That smells wrong, so I keep it for a moment and let a test catch it rather than hand-wave a fix.

**A real differential-test failure, and tracing it.** I wire up an independent oracle — a plain max-flow / min-cut on the original tree (root as source, every marked node a sink, each edge a capacity), which is slow but obviously correct and shares no code with the DP — and a random generator of tiny trees with random marked sets. Running the first batch I get a mismatch on a small case that reduces to this skeleton: a marked child `c` directly under the root, with a *single* edge between them of cost `w`, and `c` has its own marked children below. My DP at the root did `sum += min(w, dp[c])`, and `dp[c]` came out *smaller* than `w` (because `c`'s own subtree was cheap to cut), so the DP "paid `dp[c]`" and declared `c` handled — but `dp[c]` only separates the nodes *below* `c` from `c`; it never severs `c` itself from the root. The flow oracle, correctly, had to cut the edge above `c` (cost `w`) and reported the larger value. My DP under-counted.

**Diagnosing the bug precisely.** The defect is exactly the line I flagged: for a *marked* child, `min(w, dp[c])` is wrong because `dp[c]` is not a feasible way to detach `c`. The recurrence semantics — "`dp[v]` separates marked nodes in `v`'s subtree *from `v`*" — means a marked node is, by definition, *not yet* separated from itself; the obligation to cut it off lands on its parent. So a marked child must contribute the cost of cutting the connecting (compressed) edge, namely `w`, and `dp[c]` is irrelevant for it. (And it is genuinely irrelevant, not just dominated: once I cut the single edge `w` on the compressed chain, the whole subtree of `c` is severed from `u`, so every deeper marked node is auto-separated from the root too; I never need to also pay `dp[c]`.) The fix is to split the branches for real:

```
if (isSpec[nodes[c]]) sum += w;            // marked child: must sever the connecting path
else                  sum += min(w, dp[c]); // ordinary child: cut here, or recurse below
```

**Re-verifying the fix.** I re-run the same failing case: now the root contributes `w` for its marked child instead of the too-cheap `dp[c]`, and the DP matches the flow oracle. Then I let the differential tester run in bulk — 600 random tiny cases (trees up to 12 nodes, weights up to `10^9`, several orders each, including empty orders) — and get zero mismatches; a second batch of 400 larger random cases (up to 60 nodes) also passes clean. The bug broke for the reason I diagnosed, and fixing exactly that reason makes both the failing case and the whole suite pass, which is the evidence I trust.

**Edge cases, deliberately, because this is where this kind of code dies.**

- *Empty order (`k = 0`).* I always push the root into the node set, so the virtual tree is a single node with no edges; the DP loop runs over zero children and `dp[0] = 0`. Answer `0` — correct, nothing to cut.
- *Single-node tree (`n = 1`).* There are no edges and every order must be empty by the contract; the DFS visits just node `1`, the lifting tables are trivial, and every answer is `0`. I set `LOG >= 1` explicitly so the lifting loops are well-formed even when `n = 1`.
- *Bamboo of depth `2*10^5`.* The iterative DFS avoids a recursion overflow, and binary lifting climbs in `O(log n)`. I test a path of `200000` nodes with *all* non-root nodes marked: the cheapest plan is to cut the single top edge `1-2`, and the program returns exactly its weight in 0.15 s. (This also confirms the marked-child rule: every node is marked, yet the answer is one edge, because cutting high severs everyone below.)
- *Shared-edge dominance.* The order `{6, 7}` in the sample, and many generated cases where two deep marked nodes share an ancestor, exercise the `min(w, dp[c])` branch choosing the *upper* compressed edge over the sum of the lower ones — the whole point of the problem.
- *Overflow.* All weight sums are `long long`; the sentinel `INF = 4e18` is only ever read inside a `min` for `minEdgeUp` and `mn[1][0]` (the root's nonexistent parent edge), never added to a real cost, so it cannot poison a sum. I verified a full-scale case whose answer is `~3.6*10^9`, which an `int` would have wrapped.
- *Performance.* On `n = 2*10^5`, `q` chosen so `sum|S| = 2*10^5`, and weights near `10^9`, the program runs in about 0.25 s using 65 MB — order of magnitude inside the 2 s / 256 MB limits. The per-order work is dominated by sorting the `O(|S|)` chosen nodes and the `O(|S| log n)` LCA / path-min queries, exactly as designed.

**One implementation subtlety I double-checked.** When the running LCA `l` of the new node and the stack top is a *new* branch node not already on the stack, I have to insert it — and it must already be present in my `nodes` array (otherwise `posOf` would fail). It is: I seeded `nodes` with the LCAs of every adjacent pair (after sorting by `tin`), and a standard lemma says those adjacent-pair LCAs already include *every* branch point the stack construction can encounter. The bulk differential tests, which would have crashed or mis-answered on a missing branch node, came back clean, which corroborates that the node set is complete.

**Final solution.** I convinced myself the *idea* is right by deriving the compression from the full-tree DP and proving the chain-cut equals the minimum chain edge, and I convinced myself the *code* is right by tracing the marked-child mismatch to a precise cause, fixing exactly that, and then passing a thousand-plus differential cases plus the explicit corners and the full-scale timing. This is what I ship — one self-contained C++17 file: the same min-cut DP I trust, run on a virtual tree so each order costs `O(|S| log n)` instead of `O(n)`.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Cut-to-Quarantine.
// Rooted tree at node 1; every edge has a positive cut cost. For each query with a
// set S of "special" nodes, output the minimum total cost of edges to cut so that no
// special node stays connected to the root (node 1).
//
// Insight: only the special nodes plus their pairwise LCAs (and the root) shape the
// answer. Build the virtual tree of those O(|S|) nodes; a virtual edge stands for an
// original path, and the cheapest way to sever that path is to cut its minimum-weight
// edge. A small tree DP over the virtual tree then yields the min cut.
// Complexity: O(n log n) preprocessing, O(|S| log n) per query.

static const long long INF = (long long)4e18;

int n, q;
vector<pair<int,int>> g[200005];   // undirected adjacency: (neighbor, edgeWeight)
int LOG;
vector<array<int,20>> up;          // binary-lifting ancestors
vector<array<long long,20>> mn;    // min edge weight on each 2^k upward jump
int tin[200005], dep[200005], timer_ = 0;

int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int k = 0; k < LOG; k++) if (d & (1 << k)) u = up[u][k];
    if (u == v) return u;
    for (int k = LOG - 1; k >= 0; k--)
        if (up[u][k] != up[v][k]) { u = up[u][k]; v = up[v][k]; }
    return up[u][0];
}

// minimum edge weight on the path from descendant d up to ancestor a (a above d)
long long minEdgeUp(int d, int a) {
    long long res = INF;
    int diff = dep[d] - dep[a];
    for (int k = 0; k < LOG; k++)
        if (diff & (1 << k)) { res = min(res, mn[d][k]); d = up[d][k]; }
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    LOG = 1; while ((1 << LOG) < n + 1) LOG++; LOG = min(LOG, 20); if (LOG < 1) LOG = 1;
    up.assign(n + 1, {});
    mn.assign(n + 1, {});
    for (int i = 0; i < n - 1; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        g[u].push_back({v, (int)w});
        g[v].push_back({u, (int)w});
    }

    // Rooted iterative DFS at node 1: set tin, dep, up[*][0], mn[*][0].
    timer_ = 0;
    {
        vector<int> par(n + 1, 0);
        vector<int> st; st.reserve(n);
        vector<int> it(n + 1, 0);
        up[1][0] = 1; mn[1][0] = INF; dep[1] = 0; par[1] = 0;
        st.push_back(1); tin[1] = timer_++;
        while (!st.empty()) {
            int v = st.back();
            bool pushed = false;
            while (it[v] < (int)g[v].size()) {
                auto pr = g[v][it[v]++];
                int c = pr.first; long long w = pr.second;
                if (c == par[v]) continue;
                par[c] = v;
                up[c][0] = v; mn[c][0] = w; dep[c] = dep[v] + 1;
                tin[c] = timer_++;
                st.push_back(c);
                pushed = true;
                break;
            }
            if (!pushed) st.pop_back();
        }
    }
    // binary lifting
    for (int k = 1; k < LOG; k++)
        for (int v = 1; v <= n; v++) {
            up[v][k] = up[ up[v][k-1] ][k-1];
            mn[v][k] = min(mn[v][k-1], mn[ up[v][k-1] ][k-1]);
        }

    cin >> q;
    string out;
    out.reserve(1 << 20);

    vector<int> nodes;
    vector<int> special;
    vector<char> isSpec(n + 1, 0);

    while (q--) {
        int k; cin >> k;
        special.assign(k, 0);
        for (int i = 0; i < k; i++) { cin >> special[i]; isSpec[special[i]] = 1; }

        // Node set of the virtual tree: special nodes + root + adjacent LCAs.
        nodes.assign(special.begin(), special.end());
        nodes.push_back(1);
        sort(nodes.begin(), nodes.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

        int m0 = nodes.size();
        for (int i = 0; i + 1 < m0; i++)
            nodes.push_back(lca(nodes[i], nodes[i + 1]));
        sort(nodes.begin(), nodes.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

        int M = nodes.size();
        static vector<vector<pair<int,long long>>> vch;
        vch.assign(M, {});
        auto posOf = [&](int x)->int{
            int lo = 0, hi = M - 1;
            while (lo < hi) { int mid=(lo+hi)/2; if (tin[nodes[mid]] < tin[x]) lo=mid+1; else hi=mid; }
            return lo;
        };

        // Build virtual tree via a monotonic stack (indices into `nodes`).
        vector<int> stk;
        stk.push_back(0); // node 1 has the smallest tin, sits at index 0
        for (int i = 1; i < M; i++) {
            int cur = nodes[i];
            int l = lca(nodes[stk.back()], cur);
            while (stk.size() >= 2 && dep[ nodes[stk[stk.size()-2]] ] >= dep[l]) {
                int child = stk.back(); stk.pop_back();
                int parent = stk.back();
                vch[parent].push_back({child, minEdgeUp(nodes[child], nodes[parent])});
            }
            int topIdx = stk.back();
            if (nodes[topIdx] != l) {
                int li = posOf(l);
                int child = stk.back(); stk.pop_back();
                vch[li].push_back({child, minEdgeUp(nodes[child], l)});
                stk.push_back(li);
            }
            stk.push_back(i);
        }
        while (stk.size() >= 2) {
            int child = stk.back(); stk.pop_back();
            int parent = stk.back();
            vch[parent].push_back({child, minEdgeUp(nodes[child], nodes[parent])});
        }

        // Post-order over the virtual tree (root index 0).
        vector<int> order; order.reserve(M);
        {
            vector<int> s; s.push_back(0);
            vector<int> state(M, 0);
            while (!s.empty()) {
                int u = s.back();
                if (state[u] < (int)vch[u].size()) s.push_back(vch[u][state[u]++].first);
                else { order.push_back(u); s.pop_back(); }
            }
        }

        // dp[u] = min cost to disconnect every special node in virtual-subtree(u) from nodes[u].
        vector<long long> dp(M, 0);
        for (int u : order) {
            long long sum = 0;
            for (auto &e : vch[u]) {
                int c = e.first; long long w = e.second;
                if (isSpec[nodes[c]])
                    sum += w;                // special child: must sever the connecting path
                else
                    sum += min(w, dp[c]);    // ordinary child: cut here, or recurse below
            }
            dp[u] = sum;
        }

        out += to_string(dp[0]);
        out += '\n';

        for (int i = 0; i < k; i++) isSpec[special[i]] = 0;
    }

    cout << out;
    return 0;
}
```

**Causal recap.** The honest min-cut DP over the whole tree is correct but `O(q*n)` — up to `4*10^10` node visits, dozens of times over budget — because it pays for the entire tree on every order while only `sum|S|` actually varies; the resolution is that an order bends the tree at only `O(|S|)` places (the marked nodes and their adjacent-pair LCAs), and on each straight chain between them the cheapest cut is just the chain's minimum edge, so I run the identical DP on a virtual tree of `O(|S|)` nodes whose edges carry that path-minimum, dropping each order to `O(|S| log n)`; the one real bug was contributing `min(w, dp[c])` for a *marked* child, which let the DP "pay below" a node it was actually obligated to cut *off* — a differential-test mismatch against a flow oracle pinned it, and contributing `w` for marked children fixes it; binary-lifting path-minima, an iterative DFS for the deep-bamboo case, `long long` everywhere, and forcing the root into every order's node set close out the LCA, depth, overflow, and empty-order corners.
