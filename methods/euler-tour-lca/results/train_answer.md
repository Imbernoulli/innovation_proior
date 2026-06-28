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

```cpp
// Reads: "n q", then n-1 undirected tree edges (1-based), then q query pairs
// (1-based). The tree is rooted at node 0 (input node 1). Prints, one per line,
// the 1-based lowest common ancestor of each queried pair.
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<vector<int>> adj(n);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // One iterative DFS records the order in which vertices are touched: a
    // vertex is appended when first entered and again every time the walk
    // returns to it from a child. The tour has length 2n-1; alongside it we
    // keep the depth at each tour position and, per vertex, the index of its
    // first appearance. Iterative DFS so a path-shaped tree can't overflow.
    vector<int> euler;        // node touched at each tour position
    vector<int> depthAt;      // depth at each tour position
    vector<int> first(n, -1); // first tour position of each node
    euler.reserve(n > 0 ? 2 * n - 1 : 0);
    depthAt.reserve(n > 0 ? 2 * n - 1 : 0);

    if (n > 0) {
        vector<char> seen(n, 0);
        // frame: node, depth, next child index into adj[node]
        vector<array<int, 3>> stk;
        int root = 0;
        seen[root] = 1;
        first[root] = 0;
        euler.push_back(root);
        depthAt.push_back(0);
        stk.push_back({root, 0, 0});

        while (!stk.empty()) {
            int node = stk.back()[0];
            int d = stk.back()[1];
            int idx = stk.back()[2];
            const vector<int>& children = adj[node];
            bool advanced = false;
            while (idx < (int)children.size()) {
                int nxt = children[idx];
                ++idx;
                if (!seen[nxt]) {
                    stk.back()[2] = idx;
                    seen[nxt] = 1;
                    first[nxt] = (int)euler.size();
                    euler.push_back(nxt);
                    depthAt.push_back(d + 1);
                    stk.push_back({nxt, d + 1, 0});
                    advanced = true;
                    break;
                }
            }
            if (!advanced) {
                stk.back()[2] = idx;
                stk.pop_back();
                if (!stk.empty()) {
                    // returned to the parent: re-record it
                    euler.push_back(stk.back()[0]);
                    depthAt.push_back(stk.back()[1]);
                }
            }
        }
    }

    // Sparse table over depthAt that stores the *position* of the minimal
    // depth, so we can recover the Euler node there. min is idempotent, so
    // overlapping the two covering blocks is harmless and the query is O(1).
    int m = (int)depthAt.size();
    vector<int> logTable(m + 1, 0);
    for (int i = 2; i <= m; ++i) logTable[i] = logTable[i >> 1] + 1;
    int levels = (m > 0 ? logTable[m] + 1 : 1);

    vector<vector<int>> sp(levels, vector<int>(m, 0));
    for (int j = 0; j < m; ++j) sp[0][j] = j;
    for (int k = 1; k < levels; ++k) {
        int span = 1 << k;
        int half = 1 << (k - 1);
        for (int j = 0; j + span <= m; ++j) {
            int a = sp[k - 1][j];
            int b = sp[k - 1][j + half];
            sp[k][j] = (depthAt[a] <= depthAt[b]) ? a : b;
        }
    }

    auto argmin = [&](int l, int r) -> int {
        int k = logTable[r - l + 1];
        int a = sp[k][l];
        int b = sp[k][r - (1 << k) + 1];
        return (depthAt[a] <= depthAt[b]) ? a : b;
    };

    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; ++i) {
        int u, v;
        cin >> u >> v;
        --u; --v;
        int l = first[u], r = first[v];
        if (l > r) swap(l, r);
        int ans = euler[argmin(l, r)] + 1;
        out += to_string(ans);
        out.push_back('\n');
    }
    cout << out;
    return 0;
}
```
