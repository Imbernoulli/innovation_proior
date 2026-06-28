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
