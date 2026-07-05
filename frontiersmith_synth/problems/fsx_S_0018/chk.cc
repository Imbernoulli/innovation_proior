#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m, n;
vector<ll> C;                 // 1..m generator budgets
vector<vector<ll>> v, w;      // (n+1) x (m+1) thrill / power

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    m = inf.readInt();
    n = inf.readInt();

    C.assign(m + 1, 0);
    for (int j = 1; j <= m; j++) C[j] = inf.readInt();

    v.assign(n + 1, vector<ll>(m + 1, 0));
    w.assign(n + 1, vector<ll>(m + 1, 0));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) v[i][j] = inf.readInt();
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) w[i][j] = inf.readInt();

    // ---- internal baseline B: naive first-fit schedule ----
    // acts 1..n in order, each placed on the lowest-indexed platform with room.
    {
        vector<ll> rem = C;
        ll B = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (w[i][j] <= rem[j]) { rem[j] -= w[i][j]; B += v[i][j]; break; }
            }
        }
        if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

        // ---- read & validate participant schedule ----
        vector<ll> load(m + 1, 0);
        ll F = 0;
        for (int i = 1; i <= n; i++) {
            int a = ouf.readInt(0, m, "platform");
            if (a == 0) continue;
            load[a] += w[i][a];
            if (load[a] > C[a])
                quitf(_wa, "platform %d overloaded: draw %lld > budget %lld", a, load[a], C[a]);
            F += v[i][a];
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

        double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
        quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    }
    return 0;
}
