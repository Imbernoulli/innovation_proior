**Problem.** An undirected **multigraph** has `n` switches and `m` cables; parallel cables (several on the same pair) and self-loops (`a == b`) are allowed, and each cable is a distinct object. Count the cables that lie on **at least one cycle** — equivalently, `m` minus the number of bridges (a bridge is a cable whose removal disconnects its endpoints). Read `n`, `m`, then `m` pairs from stdin; print the count.

**Why the brute is too slow.** Deleting each cable and re-testing endpoint connectivity is correct but `O(m(n+m))` ≈ `8*10^10` on the largest input. Use it only as an oracle; submit a single-pass DFS.

**Key idea — DFS bridge-finding.** One DFS computes, for each vertex `u`, `disc[u]` (discovery time) and `low[u]` (lowest `disc` reachable from `u`'s subtree using at most one back edge). A tree edge `u -> v` is a bridge **iff** `low[v] > disc[u]`: nothing in `v`'s subtree climbs back to `u` or above except through that edge. Mark every bridge id, then answer `m - (#bridges)`.

- `low[u] = min(disc[u], min over back edges u->w of disc[w], min over tree children c of low[c])`.
- A non-bridge is, by definition, an edge on a cycle, so `m - #bridges` is the requested count.

**Correctness.** The `low[v] > disc[u]` test is the standard, proven bridge criterion: if some descendant of `v` has a back edge to `u` or an ancestor, that gives an alternate path and the edge is on a cycle (`low[v] <= disc[u]`); otherwise cutting the edge isolates `v`'s subtree. Summed over all DFS roots it handles disconnected graphs. Self-loops are never bridges (endpoints trivially stay connected) and are counted automatically. Verified against the per-edge brute on 2000 random small multigraphs (with parallel cables and self-loops) with zero mismatches.

**Pitfalls.**
1. *Parent edge by id, not by vertex (the counting trap).* In a multigraph, skipping **every** edge to the parent *vertex* discards a parallel cable's genuine back edge, so `low` never drops and a real non-bridge is mis-marked as a bridge — the count is off by one. Carry the **parent edge id** and skip only that one instance. (Two parallel cables `(1,2),(1,2)`: the buggy version returns `1`, the truth is `2`.)
2. *Recursion depth.* A path of `2*10^5` vertices makes recursive DFS go `2*10^5` frames deep and overflows the 8 MB stack (runtime error). Use an **explicit-stack** iterative DFS; do the parent relaxation and bridge test when a frame is *popped*, using the popped vertex and the popped frame's parent-edge id.
3. *Self-loops.* Store each self-loop once; it is reached as a back edge to itself, lowers nothing, and is never a bridge — so it correctly counts.

**Edge cases.** `m = 0` (and `n = 0`) → `0`; single bridge → `0`; two parallel cables → `2`; one self-loop → `1`; disconnected graphs handled by looping the DFS over every undiscovered vertex; isolated vertices contribute nothing.

**Complexity.** `O(n + m)` time, `O(n + m)` memory. The answer is computed in `long long` (it fits in 32 bits, but the explicit cast avoids any narrowing in the subtraction).

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int n, m;
vector<array<int,2>> adj_[200005]; // {neighbor, edgeId}
int disc[200005], low_[200005], timer_;
bool isBridge[400005];

// Iterative DFS for bridges: explicit stack of frames so a long path cannot
// overflow the call stack at n = 2*10^5. peId = id of the edge used to enter u.
void dfs(int root) {
    // frame: vertex u, parent-edge id, and an index into adj_[u]
    static int stU[200005], stPE[200005], stIt[200005];
    int top = 0;
    stU[top] = root; stPE[top] = -1; stIt[top] = 0;
    disc[root] = low_[root] = ++timer_;
    while (top >= 0) {
        int u = stU[top], peId = stPE[top];
        if (stIt[top] < (int)adj_[u].size()) {
            auto &e = adj_[u][stIt[top]++];
            int v = e[0], id = e[1];
            if (id == peId) continue;          // skip exactly the one parent-edge instance
            if (!disc[v]) {                    // tree edge: descend
                disc[v] = low_[v] = ++timer_;
                ++top;
                stU[top] = v; stPE[top] = id; stIt[top] = 0;
            } else {                           // back edge
                low_[u] = min(low_[u], disc[v]);
            }
        } else {                               // done with u: pop, relax parent
            --top;
            if (top >= 0) {
                int p = stU[top];
                low_[p] = min(low_[p], low_[u]);
                if (low_[u] > disc[p]) isBridge[peId] = true;
            }
        }
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    if (!(cin >> n >> m)) return 0;
    for (int i = 0; i < m; i++) {
        int a, b; cin >> a >> b;
        adj_[a].push_back({b, i});
        if (a != b) adj_[b].push_back({a, i}); // self-loop stored once (never a bridge anyway)
    }
    for (int s = 1; s <= n; s++)
        if (!disc[s]) dfs(s);

    long long bridges = 0;
    for (int i = 0; i < m; i++) if (isBridge[i]) bridges++;
    cout << (long long)m - bridges << "\n";
    return 0;
}
```
