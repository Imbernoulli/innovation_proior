#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> w;             // 1..n hype values
vector<pair<int,int>> edges;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    w.assign(n + 1, 0);
    ll B = 0;                       // baseline: largest single hype value
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readInt();
        B = max(B, (ll)w[i]);
    }
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n);
        int v = inf.readInt(1, n);
        edges[i] = {u, v};
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's chosen roster ----
    int c = ouf.readInt(0, n, "c");
    vector<char> chosen(n + 1, 0);
    ll F = 0;
    for (int i = 0; i < c; i++) {
        int idx = ouf.readInt(1, n, "performerIndex");
        if (chosen[idx]) quitf(_wa, "performer %d featured more than once", idx);
        chosen[idx] = 1;
        F += w[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: chosen set must be independent (no conflict inside it) ----
    for (auto& e : edges) {
        if (chosen[e.first] && chosen[e.second])
            quitf(_wa, "performers %d and %d conflict but are both featured",
                  e.first, e.second);
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
