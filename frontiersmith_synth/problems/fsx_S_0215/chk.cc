#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    vector<ll> w(n + 1);
    ll B = 0;
    for (int i = 1; i <= n; i++) {
        w[i] = inf.readInt();
        B = max(B, w[i]);
    }
    vector<vector<int>> adj(n + 1);
    for (int e = 0; e < m; e++) {
        int u = inf.readInt();
        int v = inf.readInt();
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's independent set ----
    int r = ouf.readInt(0, n, "r");
    vector<char> inSet(n + 1, 0);
    vector<int> chosen;
    chosen.reserve(r);
    for (int i = 0; i < r; i++) {
        int idx = ouf.readInt(1, n, "vertex");
        if (inSet[idx]) quitf(_wa, "individual %d monitored more than once", idx);
        inSet[idx] = 1;
        chosen.push_back(idx);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // independence check: no chosen vertex may have a chosen neighbor
    ll F = 0;
    for (int u : chosen) {
        for (int v : adj[u])
            if (inSet[v])
                quitf(_wa, "individuals %d and %d are in direct contact but both monitored", u, v);
        F += w[u];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
