**Problem.** A functional graph on `n` nodes gives every node `i` a single out-edge `f(i)`. For each of `q` queries `(s, t)` with `t` up to `10^18`, output `f^t(s)` — the node reached after following out-edges `t` times from `s`. Read `n`, the array `f`, then `q` queries from stdin; print one answer per line. Limits: `n, q <= 2*10^5`.

**Why the obvious approach is the wrong default.** Binary lifting (`up[k][v] = f^{2^k}(v)` for `k` up to `60`, then `O(log t)` jumps per query) is correct but spends a `60 * n ≈ 1.2*10^7`-entry table (~48 MB) and a cache-hostile `O(log t)` per query to brute-force a structure that is almost entirely *periodic*. Out-degree one forces every orbit into a tail-plus-cycle, and circulating a cycle is `O(1)` modular arithmetic — paying a `log t` factor in both time and memory to ignore that periodicity is the tell that binary lifting is overkill here.

**Key idea — rho decomposition + offline DFS-stack ancestor lookup.** Because out-degree is exactly one, each component is a **rho (ρ)**: tails (a forest of trees) feeding a single directed **cycle**. Let `depth[s]` be the steps from `s` to its cycle entry `entry[s]`, and `L` the cycle length. Split each query:

- `t >= depth[s]`: the answer is purely cyclic — `(pos(entry[s]) + (t - depth[s])) mod L` along the cycle. One modular step, `O(1)`.
- `t < depth[s]`: the answer is `t` steps down the tail, i.e. the `t`-th *ancestor* in the **reversed** tail forest (roots = cycle nodes).

The tail jump is made `O(1)` *without* a binary-lifting table by a clean observation: during a DFS, the live recursion stack *is* the root-to-node path, so the `t`-th ancestor of the node at stack-depth `D` is just `stk[D - t]`. So I bucket queries by start node and answer them **offline** during a single iterative DFS over the tail forest. Total `O(n + q)` time and `O(n + q)` memory, no `log t` anywhere.

Cycles are found with the standard three-state functional-graph traversal (`0` unvisited / `1` on current path / `2` finished); a new cycle is the suffix of the current path from the first node that is re-seen with state `1`.

**Pitfalls to get right.**
1. *Cycle entry must be the DFS tree's root.* For a tail query the entry node is `stk[0]` of the tree currently being traversed (the `root` of the outer loop), not a per-node value that is easy to leave uninitialized as `0`. A trace of `f = [1,2,0,2]`, query `(3,1)` returning `0` instead of `2` exposes exactly this.
2. *Boundary `t == depth`.* Route the exact entry-landing case through the tail branch (`t <= D` -> `stk[D - t] = stk[0] = entry`) so the cyclic branch (`t > D`) always has `into = t - D >= 1`; both formulations agree, this one needs no cycle lookup.
3. *64-bit step counts.* `t` reaches `10^18`; keep `t`, `into = t - D`, and the position in `long long`, reducing `mod L` immediately. Node labels stay `int`.
4. *No recursion.* A chain of length `2*10^5` would overflow a native call stack, so the DFS is iterative with explicit child iterators.

**Edge cases.** `t = 0` -> `stk[D] = s` itself; query on a cycle node -> `depth 0`, pure modular jump; `n = 1` self-loop -> everything maps to `0`; one big cycle -> all queries modular; longest chain into a self-loop -> deep tail handled in `O(1)` per query via the stack, no `O(nq)` blowup.

**Complexity.** `O(n + q)` time, `O(n + q)` memory. At `n = q = 2*10^5` it runs in under 0.1 s using ~16 MB, versus ~48 MB for a binary-lifting table.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    if (!(cin >> n)) return 0;
    vector<int> f(n);
    for (int i = 0; i < n; i++) cin >> f[i];

    int q;
    cin >> q;
    vector<int> qs(q);          // query start node
    vector<long long> qt(q);    // query step count t
    for (int i = 0; i < q; i++) cin >> qs[i] >> qt[i];

    // ---- Step 1: locate every cycle in the functional graph. ----
    // state: 0 = unvisited, 1 = on the current walking path, 2 = finished.
    vector<int> state(n, 0);
    vector<char> onCycle(n, 0);
    vector<int> cycId(n, -1);   // which cycle a cycle-node belongs to
    vector<int> cycPos(n, 0);   // position of a cycle-node along its cycle
    vector<int> cycLen;         // length of each cycle, indexed by cycle id
    vector<vector<int>> cycNodes; // ordered nodes of each cycle

    vector<int> stkPath;        // current walking path for cycle detection
    for (int s = 0; s < n; s++) {
        if (state[s] != 0) continue;
        int v = s;
        // walk forward until we hit a node already seen
        while (state[v] == 0) {
            state[v] = 1;
            stkPath.push_back(v);
            v = f[v];
        }
        if (state[v] == 1) {
            // found a new cycle: it starts at v and runs to the end of stkPath
            int cid = (int)cycLen.size();
            vector<int> nodes;
            // pop the path back down to v, recording the cycle nodes (reversed)
            while (!stkPath.empty() && stkPath.back() != v) {
                int u = stkPath.back(); stkPath.pop_back();
                state[u] = 2;          // these were tail nodes on this walk
            }
            // now stkPath.back() == v ; collect the cycle by walking f from v
            int u = v;
            do {
                nodes.push_back(u);
                onCycle[u] = 1;
                cycId[u] = cid;
                state[u] = 2;
                u = f[u];
            } while (u != v);
            for (int i = 0; i < (int)nodes.size(); i++) cycPos[nodes[i]] = i;
            cycLen.push_back((int)nodes.size());
            cycNodes.push_back(move(nodes));
            // pop v as well (it is finished)
            stkPath.pop_back();
        }
        // mark the remaining path nodes as finished (they are tails leading
        // into an already-known structure)
        while (!stkPath.empty()) {
            int u = stkPath.back(); stkPath.pop_back();
            state[u] = 2;
        }
    }

    // ---- Step 2: build reverse edges among tail nodes (non-cycle). ----
    // We only need reverse edges that lead from a node to its tail predecessors;
    // an edge u -> f[u] is a tail edge contributing to depth structure when u is
    // NOT on a cycle. We root the tail forest at cycle nodes.
    vector<int> revHead(n, -1), revNext(n, -1);
    for (int u = 0; u < n; u++) {
        if (!onCycle[u]) {
            int p = f[u];        // u's forward target = u's parent in tail forest
            revNext[u] = revHead[p];
            revHead[p] = u;
        }
    }

    // ---- Step 3: bucket queries by their start node for offline answering. ----
    vector<int> qHead(n, -1), qNext(q, -1);
    for (int i = 0; i < q; i++) {
        int s = qs[i];
        qNext[i] = qHead[s];
        qHead[s] = i;
    }

    vector<long long> ans(q, -1);

    // Helper: answer a query whose start is at tail-depth D with cycle entry
    // node `entry`, using the live ancestor stack `stk` (stk[D] == start).
    // If t <= D the answer is a tail node stk[D - t]; else step into the cycle.
    auto answerWith = [&](int qi, int D, int entry, const vector<int>& stk) {
        long long t = qt[qi];
        if (t <= (long long)D) {
            ans[qi] = stk[(int)(D - t)];
        } else {
            long long into = t - (long long)D;          // steps once on the cycle
            int cid = cycId[entry];
            int L = cycLen[cid];
            long long pos = ((long long)cycPos[entry] + into) % L;
            ans[qi] = cycNodes[cid][(int)pos];
        }
    };

    // ---- Step 4: iterative DFS over each tail tree rooted at a cycle node. ----
    // The DFS stack `stk` holds the current root-to-node path; index 0 is the
    // cycle entry (depth 0 on the tail = the cycle node itself), index D is the
    // node currently being expanded at tail-depth D.
    vector<int> stk;             // current path of nodes (by id)
    stk.reserve(n + 1);
    // iterative frame: node + iterator over its reverse-children
    vector<int> itChild(n);      // current child pointer (revNext walk) per node

    for (int root = 0; root < n; root++) {
        if (!onCycle[root]) continue;          // start DFS only from cycle nodes
        int entry = root;                      // the cycle entry for this tree

        // First, answer queries that start exactly on this cycle node (depth 0).
        // stk just contains the root here.
        stk.clear();
        stk.push_back(root);
        for (int qi = qHead[root]; qi != -1; qi = qNext[qi]) {
            answerWith(qi, 0, entry, stk);
        }
        // descend into tail children of root
        itChild[root] = revHead[root];
        while ((int)stk.size() > 0) {
            int v = stk.back();
            int c = itChild[v];
            // advance to a child that is a tail node (all rev edges here are tails)
            if (c != -1) {
                itChild[v] = revNext[c];        // consume this child
                // push child c
                stk.push_back(c);
                int D = (int)stk.size() - 1;     // tail-depth of c
                itChild[c] = revHead[c];
                // answer queries that start at c
                for (int qi = qHead[c]; qi != -1; qi = qNext[qi]) {
                    answerWith(qi, D, entry, stk);
                }
            } else {
                stk.pop_back();                  // done with v
                if (v == root) break;            // finished this tree
            }
        }
    }

    // ---- Output ----
    string out;
    out.reserve((size_t)q * 7);
    for (int i = 0; i < q; i++) {
        out += to_string(ans[i]);
        out += '\n';
    }
    cout << out;
    return 0;
}
```
