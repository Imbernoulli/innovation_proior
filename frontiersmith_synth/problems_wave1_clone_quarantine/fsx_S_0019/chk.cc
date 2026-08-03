#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Buried Cable Trenches (degree-capped Steiner-flavoured network).
// Minimization. Markers 1..T are artifact chambers that MUST end up in one connected
// component; markers T+1..N are optional relay posts (Steiner junctions). Validates:
// distinct valid endpoints, no duplicate trench, per-marker degree <= cap, and that all
// T chambers are mutually connected (posts may bridge them). Baseline B = length of the
// input-order chamber chain 1-2-...-T (always feasible, positive). Participant length F =
// sum of rounded Euclidean trench lengths.  ratio = min(1, (B / max(1,F)) / 10).

static vector<long long> X, Y;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    return (long long)llround(sqrt(dx * dx + dy * dy));
}

struct DSU {
    vector<int> p;
    DSU(int n) : p(n) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void unite(int a, int b) { p[find(a)] = find(b); }
};

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    X.resize(N);
    Y.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        cap[i] = inf.readInt();
    }

    // read participant network. E may be 0. Upper bound 2*N (sum of caps <= 4N -> <=2N edges).
    int E = ouf.readInt(0, 2 * N, "E");
    vector<int> deg(N, 0);
    set<pair<int,int>> seen;
    DSU dsu(N);
    long long F = 0;
    for (int k = 0; k < E; k++) {
        int u = ouf.readInt(1, N, "a") - 1;
        int v = ouf.readInt(1, N, "b") - 1;
        if (u == v) quitf(_wa, "trench %d is a self-loop at marker %d", k + 1, u + 1);
        int a = min(u, v), b = max(u, v);
        if (!seen.insert({a, b}).second)
            quitf(_wa, "duplicate trench between markers %d and %d", a + 1, b + 1);
        deg[u]++;
        deg[v]++;
        F += len(u, v);
        dsu.unite(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the %d trenches", E);

    for (int i = 0; i < N; i++)
        if (deg[i] > cap[i])
            quitf(_wa, "marker %d has degree %d exceeding its cap %d", i + 1, deg[i], cap[i]);

    // all artifact chambers (0..T-1) must share one component
    int root = dsu.find(0);
    for (int i = 1; i < T; i++)
        if (dsu.find(i) != root)
            quitf(_wa, "chamber %d is not connected to chamber 1", i + 1);

    if (F <= 0) quitf(_wa, "network has zero length but chambers require connection");

    // baseline: input-order chamber chain 1-2-...-T
    long long B = 0;
    for (int i = 0; i + 1 < T; i++) B += len(i, i + 1);
    if (B <= 0) quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
