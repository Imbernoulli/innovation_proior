#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll DIST(ll x1, ll y1, ll x2, ll y2) {
    ll dx = x1 - x2, dy = y1 - y2;
    return (ll)llround(sqrt((double)(dx * dx + dy * dy)));
}

// 20-bit-per-axis Morton (Z-order) key; coords are in [0,1000000] < 2^20.
static inline unsigned long long mortonKey(unsigned int x, unsigned int y) {
    unsigned long long k = 0;
    for (int i = 0; i < 20; i++) {
        k |= ((unsigned long long)((x >> i) & 1u)) << (2 * i);
        k |= ((unsigned long long)((y >> i) & 1u)) << (2 * i + 1);
    }
    return k;
}

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { iota(p.begin(), p.end(), 0); }
    int find(int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }
    void unite(int a, int b) { p[find(a)] = find(b); }
};

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    vector<ll> X(n), Y(n);
    vector<int> B(n);
    for (int i = 0; i < n; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        B[i] = inf.readInt();
    }

    // ---- internal baseline B: Morton-ordered chain ----
    vector<int> ord(n);
    iota(ord.begin(), ord.end(), 0);
    sort(ord.begin(), ord.end(), [&](int a, int b) {
        unsigned long long ka = mortonKey((unsigned)X[a], (unsigned)Y[a]);
        unsigned long long kb = mortonKey((unsigned)X[b], (unsigned)Y[b]);
        if (ka != kb) return ka < kb;
        return a < b;
    });
    ll Bcost = 0;
    for (int i = 1; i < n; i++)
        Bcost += DIST(X[ord[i-1]], Y[ord[i-1]], X[ord[i]], Y[ord[i]]);
    if (Bcost < 1) Bcost = 1;

    // ---- read & validate participant backbone ----
    int m = ouf.readInt(0, 3 * n, "m");
    vector<int> deg(n, 0);
    DSU dsu(n);
    ll F = 0;
    set<pair<int,int>> seen;
    for (int e = 0; e < m; e++) {
        int u = ouf.readInt(1, n, "u");
        int v = ouf.readInt(1, n, "v");
        if (u == v) quitf(_wa, "self-loop at node %d", u);
        u--; v--;
        int a = min(u, v), b2 = max(u, v);
        if (seen.count({a, b2})) quitf(_wa, "duplicate link {%d,%d}", a + 1, b2 + 1);
        seen.insert({a, b2});
        deg[u]++; deg[v]++;
        if (deg[u] > B[u]) quitf(_wa, "contact cap exceeded at node %d", u + 1);
        if (deg[v] > B[v]) quitf(_wa, "contact cap exceeded at node %d", v + 1);
        F += DIST(X[u], Y[u], X[v], Y[v]);
        dsu.unite(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output");

    // ---- connectivity: all nodes in one component ----
    int r = dsu.find(0);
    for (int i = 1; i < n; i++)
        if (dsu.find(i) != r) quitf(_wa, "backbone not connected (node %d isolated)", i + 1);

    if (F < 1) F = 1;
    double sc = min(1000.0, 100.0 * (double)Bcost / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bcost, sc / 1000.0);
    return 0;
}
