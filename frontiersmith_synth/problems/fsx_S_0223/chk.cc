#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n;
vector<ll> X, Y, C;

static inline ll len(int a, int b) {
    ll dx = X[a] - X[b], dy = Y[a] - Y[b];
    return (ll)llround(sqrt((double)(dx * dx + dy * dy)));
}

struct DSU {
    vector<int> p;
    DSU(int m) : p(m) { for (int i = 0; i < m; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    bool uni(int a, int b) { a = find(a); b = find(b); if (a == b) return false; p[a] = b; return true; }
};

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    X.assign(n + 1, 0); Y.assign(n + 1, 0); C.assign(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        C[i] = inf.readInt();
    }

    // internal baseline B: total length of the input-order chain 1-2-...-n
    ll B = 0;
    for (int i = 1; i < n; i++) B += len(i, i + 1);
    if (B <= 0) B = 1; // degenerate coincident points; keep baseline positive

    // ---- read & validate participant's link set ----
    int E = ouf.readInt(n - 1, 4 * n, "E");
    vector<int> deg(n + 1, 0);
    DSU dsu(n + 1);
    set<pair<int,int>> seen;
    ll F = 0;
    for (int e = 0; e < E; e++) {
        int u = ouf.readInt(1, n, "u");
        int v = ouf.readInt(1, n, "v");
        if (u == v) quitf(_wa, "self-loop at node %d", u);
        int a = min(u, v), b = max(u, v);
        if (!seen.insert({a, b}).second) quitf(_wa, "duplicate link {%d,%d}", a, b);
        deg[u]++; deg[v]++;
        if (deg[u] > C[u]) quitf(_wa, "node %d exceeds capacity %lld (degree %d)", u, C[u], deg[u]);
        if (deg[v] > C[v]) quitf(_wa, "node %d exceeds capacity %lld (degree %d)", v, C[v], deg[v]);
        dsu.uni(u, v);
        F += len(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // connectivity: all nodes in one component with node 1
    int root = dsu.find(1);
    for (int i = 2; i <= n; i++)
        if (dsu.find(i) != root)
            quitf(_wa, "network disconnected: node %d not reachable from node 1", i);

    if (F <= 0) F = 0; // let max(1,F) guard handle it
    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
