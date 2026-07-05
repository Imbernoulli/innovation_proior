#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int m = inf.readInt();
    int n = inf.readInt();
    vector<ll> c(m + 1);
    for (int i = 1; i <= m; i++) c[i] = inf.readInt();
    vector<ll> w(n + 1);
    vector<vector<ll>> e(m + 1, vector<ll>(n + 1));
    for (int j = 1; j <= n; j++) {
        w[j] = inf.readInt();
        for (int i = 1; i <= m; i++) e[i][j] = inf.readInt();
    }

    // ---- internal baseline B: first-fit in input order (lowest-index drone) ----
    vector<ll> rem = c;
    ll B = 0;
    for (int j = 1; j <= n; j++) {
        for (int i = 1; i <= m; i++) {
            if (rem[i] >= e[i][j]) { rem[i] -= e[i][j]; B += w[j]; break; }
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant assignment ----
    vector<ll> load(m + 1, 0);
    ll F = 0;
    for (int j = 1; j <= n; j++) {
        int a = ouf.readInt(0, m, format("a_%d", j).c_str());
        if (a >= 1) { load[a] += e[a][j]; F += w[j]; }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int i = 1; i <= m; i++) {
        if (load[i] > c[i])
            quitf(_wa, "drone %d over budget: used %lld > capacity %lld", i, load[i], c[i]);
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
