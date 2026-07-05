#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for the Solar Farm DC Collection Grid (port-capped, fee-weighted, L1).
// Minimization.  Validates: distinct valid endpoints, no duplicate cable run, per-inverter
// degree <= cap, and connectivity spanning all N inverters.  Objective
//   F = sum over runs of L1 trench length  +  sum over inverters of w_i * deg(i).
// Baseline B = cost of the input-order chain 1-2-...-N (always feasible, positive).
//   ratio = min(1, (B / max(1,F)) / 10).

static vector<long long> X, Y;
static inline long long trench(int a, int b) {
    return llabs(X[a] - X[b]) + llabs(Y[a] - Y[b]);
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
    X.assign(N, 0);
    Y.assign(N, 0);
    vector<long long> w(N);
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        X[i] = inf.readInt();
        Y[i] = inf.readInt();
        cap[i] = inf.readInt();
        w[i] = inf.readInt();
    }

    // ---- read participant network ----
    // At least N-1 runs are required to connect N inverters; at most sum(cap)/2 useful runs,
    // bounded loosely by 4*N (cap<=4).
    int E = ouf.readInt(N - 1, 4 * N, "E");
    vector<int> deg(N, 0);
    DSU dsu(N);
    set<pair<int,int>> seen;
    long long trenchTotal = 0;

    for (int e = 0; e < E; e++) {
        int a = ouf.readInt(1, N, "a") - 1;
        int b = ouf.readInt(1, N, "b") - 1;
        if (a == b) quitf(_wa, "run %d joins inverter %d to itself", e, a + 1);
        pair<int,int> key = minmax(a, b);
        if (!seen.insert(key).second)
            quitf(_wa, "duplicate cable run between inverters %d and %d", a + 1, b + 1);
        deg[a]++; deg[b]++;
        if (deg[a] > cap[a]) quitf(_wa, "inverter %d exceeds port cap %d", a + 1, cap[a]);
        if (deg[b] > cap[b]) quitf(_wa, "inverter %d exceeds port cap %d", b + 1, cap[b]);
        dsu.unite(a, b);
        trenchTotal += trench(a, b);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the cable runs");

    // ---- connectivity ----
    int root = dsu.find(0);
    for (int i = 1; i < N; i++)
        if (dsu.find(i) != root)
            quitf(_wa, "network is not connected (inverter %d unreachable)", i + 1);

    // ---- participant objective F ----
    long long feeTotal = 0;
    for (int i = 0; i < N; i++) feeTotal += w[i] * (long long)deg[i];
    long long F = trenchTotal + feeTotal;

    // ---- baseline B: input-order chain 1-2-...-N ----
    long long Btrench = 0;
    for (int i = 0; i + 1 < N; i++) Btrench += trench(i, i + 1);
    long long Bfee = 0;
    for (int i = 0; i < N; i++) {
        int d = (i == 0 || i == N - 1) ? 1 : 2;  // ends degree 1, interior degree 2
        Bfee += w[i] * (long long)d;
    }
    long long B = Btrench + Bfee;
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
