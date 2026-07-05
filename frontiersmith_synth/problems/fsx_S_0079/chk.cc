#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> eu, ev; vector<ll> ew;
vector<int> cap;

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n + 1), r(n + 1, 0) { for (int i = 0; i <= n; i++) p[i] = i; }
    int f(int x) { while (p[x] != x) x = p[x] = p[p[x]]; return x; }
    bool u(int a, int b) { a = f(a); b = f(b); if (a == b) return false;
        if (r[a] < r[b]) swap(a, b); p[b] = a; if (r[a] == r[b]) r[a]++; return true; }
};

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    cap.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) cap[v] = inf.readInt();

    eu.assign(m + 1, 0); ev.assign(m + 1, 0); ew.assign(m + 1, 0);
    // cheapest direct cable between consecutive stations -> daisy-chain baseline
    vector<ll> chain(n + 1, LLONG_MAX);
    for (int i = 1; i <= m; i++) {
        int u = inf.readInt(), v = inf.readInt(); ll w = inf.readInt();
        eu[i] = u; ev[i] = v; ew[i] = w;
        int lo = min(u, v), hi = max(u, v);
        if (hi == lo + 1) chain[lo] = min(chain[lo], w);
    }

    ll B = 0;
    for (int i = 1; i < n; i++) {
        if (chain[i] == LLONG_MAX) quitf(_fail, "bad instance: no cable between %d and %d", i, i + 1);
        B += chain[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's installed cable set ----
    int r = ouf.readInt(0, m, "r");
    vector<char> usedEdge(m + 1, 0);
    vector<int> deg(n + 1, 0);
    DSU dsu(n);
    ll F = 0;
    int comps = n;
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, m, "cableIndex");
        if (usedEdge[idx]) quitf(_wa, "cable %d installed more than once", idx);
        usedEdge[idx] = 1;
        int u = eu[idx], v = ev[idx];
        deg[u]++; deg[v]++;
        if (deg[u] > cap[u]) quitf(_wa, "station %d exceeds its %d-port limit", u, cap[u]);
        if (deg[v] > cap[v]) quitf(_wa, "station %d exceeds its %d-port limit", v, cap[v]);
        if (dsu.u(u, v)) comps--;
        F += ew[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (comps != 1) quitf(_wa, "installed fabric is not connected (%d components)", comps);
    if (F <= 0) quitf(_wa, "empty / zero-length fabric");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
