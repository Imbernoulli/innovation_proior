#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Luminet (degree-capped smart-city lighting Steiner network).
// Minimization. Validates: distinct valid endpoints, no duplicate segment, per-location
// port degree <= cap, and connectivity of ALL light poles (terminals) in one component
// (junction cabinets may be unused). Baseline B = length of the pole-chain in input index
// order (always feasible, positive). Participant length F = sum of rounded Euclidean
// segment lengths. ratio = min(1, (B / max(1,F)) / 10).

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
    X.resize(N);
    Y.resize(N);
    vector<int> term(N), cap(N);
    vector<int> poles;
    for (int i = 0; i < N; i++) {
        X[i]    = inf.readInt();
        Y[i]    = inf.readInt();
        term[i] = inf.readInt();
        cap[i]  = inf.readInt();
        if (term[i]) poles.push_back(i);
    }

    // read participant layout
    long long Emax = 5LL * N + 10;
    int E = (int)ouf.readInt(0LL, Emax, "E");
    vector<int> deg(N, 0);
    set<pair<int,int>> seen;
    DSU dsu(N);
    long long F = 0;
    for (int k = 0; k < E; k++) {
        int u = ouf.readInt(1, N, "a") - 1;
        int v = ouf.readInt(1, N, "b") - 1;
        if (u == v) quitf(_wa, "segment %d is a self-loop at location %d", k + 1, u + 1);
        int a = min(u, v), b = max(u, v);
        if (!seen.insert({a, b}).second)
            quitf(_wa, "duplicate segment between locations %d and %d", a + 1, b + 1);
        deg[u]++;
        deg[v]++;
        F += len(u, v);
        dsu.unite(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the %d segments", E);

    for (int i = 0; i < N; i++)
        if (deg[i] > cap[i])
            quitf(_wa, "location %d has degree %d exceeding its port cap %d",
                  i + 1, deg[i], cap[i]);

    // connectivity: all light poles must share one component
    int root = dsu.find(poles[0]);
    for (size_t j = 1; j < poles.size(); j++)
        if (dsu.find(poles[j]) != root)
            quitf(_wa, "light pole %d is not connected to light pole %d",
                  poles[j] + 1, poles[0] + 1);

    // baseline: pole-chain in input index order
    long long B = 0;
    for (size_t j = 0; j + 1 < poles.size(); j++)
        B += len(poles[j], poles[j + 1]);
    if (B <= 0) quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
