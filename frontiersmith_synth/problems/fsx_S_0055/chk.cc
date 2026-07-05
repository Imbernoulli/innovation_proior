#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for the Ridgeline Lift Grid (tower-capped connected ski-lift network).
// Minimization. Validates: distinct valid endpoints, no duplicate lift, per-station
// degree <= cap, and connectivity spanning all N stations. Baseline B = length of the
// input-order chain 1-2-...-N (always feasible, positive). Participant length F = sum of
// rounded lift lengths len(p,q) = round( sqrt(dx^2+dy^2) + 3*|dh| ).

static vector<long long> X, Y, Hh;
static inline long long len(int a, int b) {
    double dx = (double)(X[a] - X[b]);
    double dy = (double)(Y[a] - Y[b]);
    double dh = (double)(Hh[a] - Hh[b]);
    double c = sqrt(dx * dx + dy * dy) + 3.0 * fabs(dh);
    return (long long)llround(c);
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
    Hh.resize(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        X[i]  = inf.readInt();
        Y[i]  = inf.readInt();
        Hh[i] = inf.readInt();
        cap[i] = inf.readInt();
    }

    // read participant network. A degree-capped connected network needs >= N-1 lifts and
    // can have at most sum(cap)/2 <= 2N lifts; 8*N is a safe generous upper bound.
    long long maxE = (long long)8 * N + 1;
    int E = ouf.readInt(N > 1 ? N - 1 : 0, (int)maxE, "E");
    vector<int> deg(N, 0);
    set<pair<int,int>> seen;
    DSU dsu(N);
    long long F = 0;
    for (int k = 0; k < E; k++) {
        int u = ouf.readInt(1, N, "a") - 1;
        int v = ouf.readInt(1, N, "b") - 1;
        if (u == v) quitf(_wa, "lift %d is a self-loop at station %d", k + 1, u + 1);
        int a = min(u, v), b = max(u, v);
        if (!seen.insert({a, b}).second)
            quitf(_wa, "duplicate lift between stations %d and %d", a + 1, b + 1);
        deg[u]++;
        deg[v]++;
        F += len(u, v);
        dsu.unite(u, v);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the %d lifts", E);

    for (int i = 0; i < N; i++)
        if (deg[i] > cap[i])
            quitf(_wa, "station %d has degree %d exceeding its tower cap %d", i + 1, deg[i], cap[i]);

    if (N > 1) {
        int root = dsu.find(0);
        for (int i = 1; i < N; i++)
            if (dsu.find(i) != root)
                quitf(_wa, "network is not connected (station %d unreachable from station 1)", i + 1);
    }

    // baseline: input-order chain 1-2-...-N
    long long B = 0;
    for (int i = 0; i + 1 < N; i++) B += len(i, i + 1);
    if (B <= 0) quitf(_fail, "internal: non-positive baseline B=%lld", B);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
