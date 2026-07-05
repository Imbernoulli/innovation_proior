#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<ll> w(n + 1);
    ll B = 0;                       // baseline: value of the single most valuable site
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readLong();
        if (w[i] > B) B = w[i];
    }
    vector<pair<int,int>> edges(m);
    for (int e = 0; e < m; e++) {
        int u = inf.readInt();
        int v = inf.readInt();
        edges[e] = {u, v};
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant selection ----
    int s = ouf.readInt(0, n, "s");
    vector<char> sel(n + 1, 0);
    ll F = 0;
    for (int i = 0; i < s; i++) {
        int v = ouf.readInt(1, n, "site");
        if (sel[v]) quitf(_wa, "site %d selected more than once", v);
        sel[v] = 1;
        F += w[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d site ids)", s);

    // independence check
    for (auto& pr : edges) {
        if (sel[pr.first] && sel[pr.second])
            quitf(_wa, "infeasible: sites %d and %d conflict but both selected", pr.first, pr.second);
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
