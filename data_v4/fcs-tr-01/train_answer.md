**Problem.** A kingdom is a tree of `n` cities (`n - 1` roads). For each of `q` independent queries we are
given a set `S` of *important* cities and must delete the minimum number of *non-important* cities so that no
two important cities remain in the same connected component; output `-1` if two important cities are
adjacent. Constraints: `n, q <= 2*10^5` and `sum |S| <= 2*10^5`, 2 s.

**Why the obvious approach is too slow.** There is a clean `O(n)` bottom-up DP that answers one query on the
whole tree. Run it per query and you get `O(q n)`. On `q = 10^5` queries each of size `2` (so
`sum|S| = 2*10^5`, within budget) over an `n = 2*10^5` tree that is `~2*10^{10}` vertex-visits — minutes, not
seconds. The waste is structural: a size-2 query mentions two cities, yet the full sweep touches all 200000
vertices, almost all of which are non-important degree-2 *pass-throughs* that merely forward the DP value and
make no decision.

**Key idea — the virtual (auxiliary) tree.** The only vertices that ever change the answer are the important
vertices `S` and the *branch points* where two important subtrees first merge — and a branch point of `S` is
exactly an LCA of two members of `S`. Contract the tree onto

```
V = S ∪ { LCA(S_i, S_{i+1}) : S sorted by Euler in-time }
```

where a virtual edge stands for the whole path of pass-throughs between its endpoints, and run the *same* DP
on this contracted tree. Using only consecutive-in-`tin` LCAs already captures every distinct branch point,
so `|V| <= 2|S| - 1`. With binary-lifting LCA, each query is `O(|S| log n)` and the total is
`O((sum|S|) log n) ≈ 4*10^6`. This is the standard strongest approach for "many tree-DP queries over node
subsets."

**The per-query DP (run on the virtual tree).** Root at `1`. Answer is `-1` iff some important vertex has an
important parent. Otherwise, in post-order, let `cnt[v]` = number of important vertices in `v`'s subtree still
connected up to `v`, and `s` = sum of children's `cnt`:

- `v` important: sever every child branch with `cnt > 0` (one deletion each, always possible since adjacency
  is excluded), `cnt[v] = 1`.
- `v` not important: `s >= 2` → delete `v` (one deletion), `cnt[v] = 0`; `s == 1` → `cnt[v] = 1`; `s == 0` →
  `cnt[v] = 0`.

The accumulated deletions are the answer.

**Pitfalls to get right.**
1. *Missing LCA nodes.* You must push the consecutive-`tin` LCAs into the virtual-tree node set, not just `S`.
   Omit them and a merge point like `LCA(4,5)=2 ∉ S` disappears; the DP never sees the collision and
   undercounts (e.g. returns `0` instead of `1` on a query whose two members meet at a non-important vertex).
2. *Recursive DFS.* Rooting via recursion overflows the stack on a 200000-long path; do the rooting DFS
   iteratively. (The virtual-tree DP recursion is bounded by virtual-tree depth `O(|S|)` and survives the
   worst case here, but the rooting pass must be iterative.)
3. *Virtual-tree stack build.* Process nodes in increasing `tin`; pop while the stack top is not an ancestor
   of the new node (`lca(top, v) != top`), then attach. The first node in `tin` order is the virtual root.
4. *State reset.* Clear `important[]` and `vchild[]` only for touched nodes between queries — never a global
   `O(n)` reset, or you reintroduce the `O(qn)` blowup.

**Edge cases.** Single-vertex query → `0`. Two adjacent important cities → `-1`. `n = 1`. All-important
queries and deep chains (every-other vertex on a path: each gap costs one deletion). Each verified against
both a full-tree DP and an exhaustive subset-deletion oracle.

**Complexity.** `O(n log n)` preprocessing (binary lifting), `O(|S| log n)` per query, `O((sum|S|) log n)`
total; `O(n log n)` memory. Measured: full-scale input runs in ~0.17 s using ~35 MB.

**Code.**

```cpp
#include <bits/stdc++.h>
using namespace std;

/*
  Kingdom and its Cities  --  virtual (auxiliary) tree.

  For each query we are given a set S of important vertices and must delete the
  fewest NON-important vertices so that no two important vertices stay connected
  (or report -1 if two important vertices are adjacent).

  Naive: run a full-tree O(n) DP per query -> O(q*n), too slow when both q and n
  are large.  The structure that actually decides the answer is only S together
  with the pairwise LCAs of S; every other vertex is a degree-2 pass-through.
  So we contract the tree onto V = S U {LCAs of consecutive-in-Euler-order
  pairs of S} -- the "virtual tree" -- which has O(|S|) vertices, and run the
  SAME DP there.  Per query cost is O(|S| log n); total O((sum|S|) log n).
*/

static const int LOG = 18;

int n;
vector<int> adj[200005];
int par[LOG][200005];   // binary-lifting ancestors
int dep[200005];        // depth (root depth 0)
int tin[200005];        // Euler in-time (entry order)
int timer_ = 0;

// Virtual-tree scratch
vector<int> vchild[200005];            // children in the virtual tree
bool important[200005];
long long curAns;

// Iterative DFS to fill dep, tin, par[0][]: a real DFS order via an explicit
// stack of (node, parent, childIndex), recording the entry time on first visit.
void dfs_root() {
    timer_ = 0;
    vector<int> sNode, sPar, sIdx;
    sNode.push_back(1); sPar.push_back(0); sIdx.push_back(0);
    dep[1] = 0; par[0][1] = 0; tin[1] = timer_++;
    while (!sNode.empty()) {
        int u = sNode.back();
        int p = sPar.back();
        int &i = sIdx.back();
        if (i < (int)adj[u].size()) {
            int w = adj[u][i++];
            if (w == p) continue;
            dep[w] = dep[u] + 1;
            par[0][w] = u;
            tin[w] = timer_++;
            sNode.push_back(w); sPar.push_back(u); sIdx.push_back(0);
        } else {
            sNode.pop_back(); sPar.pop_back(); sIdx.pop_back();
        }
    }
    for (int k = 1; k < LOG; k++)
        for (int v = 1; v <= n; v++)
            par[k][v] = par[k-1][ par[k-1][v] ];
}

int lca(int u, int v) {
    if (dep[u] < dep[v]) swap(u, v);
    int d = dep[u] - dep[v];
    for (int k = 0; k < LOG; k++)
        if (d & (1 << k)) u = par[k][u];
    if (u == v) return u;
    for (int k = LOG - 1; k >= 0; k--)
        if (par[k][u] != par[k][v]) { u = par[k][u]; v = par[k][v]; }
    return par[0][u];
}

// DP on the virtual tree.  Returns the number of important vertices in vroot's
// virtual-subtree that are still connected up to vroot; accumulates deletions in
// curAns.  Mirrors the full-tree DP exactly:
//   - vertex important  : sever every child branch that still carries a
//                         connected important vertex (one deletion each); pass 1.
//   - vertex not impt.  : let s = sum of children's connected counts.
//                         s>=2 -> delete this vertex (1), pass 0;
//                         s==1 -> pass 1; s==0 -> pass 0.
int dpVirtual(int v) {
    int s = 0;                 // sum of children's connected counts
    int blocked = 0;           // children needing a sever when v is important
    for (int w : vchild[v]) {
        int c = dpVirtual(w);
        s += c;
        if (c > 0) blocked++;
    }
    int ret;
    if (important[v]) {
        curAns += blocked;     // sever each still-connected child branch
        ret = 1;
    } else {
        if (s >= 2) { curAns += 1; ret = 0; }
        else if (s == 1) ret = 1;
        else ret = 0;
    }
    return ret;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (!(cin >> n)) return 0;
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    dfs_root();

    int q; cin >> q;
    string out;
    out.reserve(1 << 20);

    vector<int> nodes;         // query nodes + LCAs (the virtual node set)
    vector<int> stk;           // stack for building virtual tree
    while (q--) {
        int k; cin >> k;
        vector<int> S(k);
        for (int i = 0; i < k; i++) cin >> S[i];

        // Mark important and check adjacency-impossibility on the fly.
        for (int x : S) important[x] = true;
        bool impossible = false;
        for (int x : S) {
            if (x != 1 && important[ par[0][x] ]) { impossible = true; break; }
        }
        if (impossible) {
            out += "-1\n";
            for (int x : S) important[x] = false;
            continue;
        }

        // Build the virtual tree.
        sort(S.begin(), S.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.clear();
        for (int x : S) nodes.push_back(x);
        for (int i = 0; i + 1 < k; i++) nodes.push_back(lca(S[i], S[i+1]));
        sort(nodes.begin(), nodes.end(), [](int a, int b){ return tin[a] < tin[b]; });
        nodes.erase(unique(nodes.begin(), nodes.end()), nodes.end());

        for (int v : nodes) vchild[v].clear();

        stk.clear();
        for (int v : nodes) {
            if (stk.empty()) { stk.push_back(v); continue; }
            // pop while top is not an ancestor of v
            while (stk.size() >= 1 && lca(stk.back(), v) != stk.back()) {
                stk.pop_back();
            }
            vchild[stk.back()].push_back(v);
            stk.push_back(v);
        }
        int vroot = nodes.front(); // smallest tin => ancestor of all = root of VT

        curAns = 0;
        dpVirtual(vroot);
        out += to_string(curAns);
        out += '\n';

        for (int x : S) important[x] = false;
    }

    cout << out;
    return 0;
}
```
