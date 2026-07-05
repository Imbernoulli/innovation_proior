#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { for (int i = 0; i <= n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void uni(int a, int b) { p[find(a)] = find(b); }
};

static inline long long keyOf(int a, int b, int N) {
    if (a > b) swap(a, b);
    return (long long)a * (N + 1) + b;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    // ---- read input ----
    int N = inf.readInt();
    int M = inf.readInt();
    vector<int> cap(N + 1);
    for (int i = 1; i <= N; i++) cap[i] = inf.readInt();

    unordered_map<long long, long long> cost;
    cost.reserve(M * 2 + 16);
    for (int k = 0; k < M; k++) {
        int u = inf.readInt();
        int v = inf.readInt();
        long long w = inf.readInt();
        cost[keyOf(u, v, N)] = w;
    }

    // ---- checker's internal baseline B: the guaranteed chain 1-2-...-N ----
    long long B = 0;
    for (int i = 1; i <= N - 1; i++) {
        auto it = cost.find(keyOf(i, i + 1, N));
        if (it == cost.end()) quitf(_fail, "internal: chain link (%d,%d) missing", i, i + 1);
        B += it->second;
    }
    if (B <= 0) quitf(_fail, "internal: non-positive baseline B=%lld", B);

    // ---- read participant mesh ----
    int E = ouf.readInt(0, M, "E");
    vector<int> deg(N + 1, 0);
    set<long long> seen;
    DSU dsu(N);
    long long F = 0;

    for (int e = 0; e < E; e++) {
        int a = ouf.readInt(1, N, "a");
        int b = ouf.readInt(1, N, "b");
        if (a == b) quitf(_wa, "link %d joins node %d to itself", e + 1, a);
        long long kk = keyOf(a, b, N);
        auto it = cost.find(kk);
        if (it == cost.end())
            quitf(_wa, "link %d (%d,%d) is not a buildable link", e + 1, a, b);
        if (seen.count(kk))
            quitf(_wa, "link %d (%d,%d) is a duplicate", e + 1, a, b);
        seen.insert(kk);
        deg[a]++; deg[b]++;
        F += it->second;
        dsu.uni(a, b);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d links", E);

    // ---- feasibility: caps ----
    for (int i = 1; i <= N; i++) {
        if (deg[i] > cap[i])
            quitf(_wa, "node %d has degree %d > cap %d", i, deg[i], cap[i]);
    }

    // ---- feasibility: connectivity over ALL N nodes ----
    int root = dsu.find(1);
    for (int i = 2; i <= N; i++) {
        if (dsu.find(i) != root)
            quitf(_wa, "mesh is not connected: node %d not reachable from node 1", i);
    }

    if (F <= 0) quitf(_wa, "empty / zero-cost mesh cannot connect %d nodes", N);

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
