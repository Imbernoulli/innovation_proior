#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int tid = atoi(argv[1]);

    // ---- scale ladder: tiny at tid=1, ~400 at tid=10 ----
    int N = 10 + tid * tid * 4;
    if (N > 400) N = 400;
    if (N < 3) N = 3;

    // ---- build a connected river tree (a "river with tributaries") ----
    // parent of node i is chosen from a recent window -> caterpillar-like branching
    int window = 1 + (tid % 5);          // vary branchiness across tests
    vector<pair<int,int>> edges;
    for (int i = 2; i <= N; i++) {
        int lo = max(1, i - 1 - (window - 1));
        int p = rnd.next(lo, i - 1);
        edges.push_back({p, i});
    }
    shuffle(edges.begin(), edges.end());

    // ---- radii: mostly small, a few modest hubs (keeps balls from covering everything) ----
    vector<int> r(N + 1);
    for (int v = 1; v <= N; v++) {
        int rr = rnd.wnext(0, 2, 2);     // heavy skew to 0/1, max 2
        r[v] = rr;
    }
    // a sparse set of reach-3 hubs so covering pays off, but not so many it trivializes
    int hubs = 1 + N / 60;
    for (int h = 0; h < hubs; h++) {
        int v = rnd.next(1, N);
        r[v] = 3;
    }

    // ---- costs: reach costs superlinearly more (tight facility-location trade-off) ----
    // c = base * (1+r)^2  keeps a high-reach ladder only a MODEST win over many small ones.
    vector<int> c(N + 1);
    for (int v = 1; v <= N; v++) {
        int base = rnd.next(2, 12);
        int val = base * (1 + r[v]) * (1 + r[v]);   // up to 12*16 = 192
        if (val < 1) val = 1;
        if (val > 200) val = 200;
        c[v] = val;
    }

    // ---- spawning grounds: a random subset, clustered a bit ----
    // pick a target fraction, ensure >=1
    double frac = 0.35 + 0.25 * rnd.next(0.0, 1.0);   // 0.35 .. 0.60
    vector<int> dem;
    vector<char> isd(N + 1, 0);
    for (int v = 1; v <= N; v++) {
        if (rnd.next(0.0, 1.0) < frac) { dem.push_back(v); isd[v] = 1; }
    }
    if (dem.empty()) { int v = rnd.next(1, N); dem.push_back(v); isd[v] = 1; }
    shuffle(dem.begin(), dem.end());

    // ---- emit ----
    printf("%d %d\n", N, N - 1);
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    for (int v = 1; v <= N; v++) printf("%d%c", c[v], v == N ? '\n' : ' ');
    for (int v = 1; v <= N; v++) printf("%d%c", r[v], v == N ? '\n' : ' ');
    printf("%d\n", (int)dem.size());
    for (size_t i = 0; i < dem.size(); i++)
        printf("%d%c", dem[i], i + 1 == dem.size() ? '\n' : ' ');
    return 0;
}
