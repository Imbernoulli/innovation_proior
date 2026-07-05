#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, r;
    DSU(int n): p(n), r(n, 0) { for (int i = 0; i < n; i++) p[i] = i; }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void uni(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
    }
};

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int K = inf.readInt();
    vector<int> eu(M + 1), ev(M + 1);
    vector<long long> ew(M + 1);
    for (int k = 1; k <= M; k++) {
        eu[k] = inf.readInt();
        ev[k] = inf.readInt();
        ew[k] = inf.readInt();
    }
    vector<int> huts(K);
    for (int i = 0; i < K; i++) huts[i] = inf.readInt();

    // Internal baseline B: cost of the whole backbone (trenches 1..N-1).
    long long B = 0;
    for (int k = 1; k <= N - 1; k++) B += ew[k];
    if (B <= 0) B = 1;  // safety; generator guarantees positive backbone

    // Read participant output.
    int E = ouf.readInt(0, M, "E");
    vector<char> used(M + 1, 0);
    DSU dsu(N + 1);
    long long F = 0;
    for (int i = 0; i < E; i++) {
        int idx = ouf.readInt(1, M, "trench_index");
        if (used[idx]) quitf(_wa, "trench index %d dug more than once", idx);
        used[idx] = 1;
        F += ew[idx];
        dsu.uni(eu[idx], ev[idx]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the trench list");

    // Feasibility: all huts in one connected component.
    int root = dsu.find(huts[0]);
    for (int i = 1; i < K; i++) {
        if (dsu.find(huts[i]) != root)
            quitf(_wa, "research huts %d and %d are not connected", huts[0], huts[i]);
    }

    long long denom = max(1LL, F);
    double sc = min(1000.0, 100.0 * (double)B / (double)denom);
    double ratio = sc / 1000.0;
    quitp(ratio, "OK F=%lld B=%lld Ratio: %.6f", F, B, ratio);
    return 0;
}
