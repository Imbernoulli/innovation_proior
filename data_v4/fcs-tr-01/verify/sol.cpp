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
