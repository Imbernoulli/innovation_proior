#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int m, n;
vector<int> cap;                 // cap[i]
vector<vector<int>> v, w;        // v[i][j], w[i][j], 1-indexed

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    m = inf.readInt();
    n = inf.readInt();
    cap.assign(m + 1, 0);
    for (int i = 1; i <= m; i++) cap[i] = inf.readInt();

    v.assign(m + 1, vector<int>(n + 1, 0));
    w.assign(m + 1, vector<int>(n + 1, 0));
    for (int j = 1; j <= n; j++) {
        for (int i = 1; i <= m; i++) {
            v[i][j] = inf.readInt();
            w[i][j] = inf.readInt();
        }
    }

    // ---- internal baseline B: deterministic first-fit assignment ----
    // stations in order 1..m; for each, scan beacons 1..n; assign an unassigned
    // beacon if it still fits the station's remaining budget.
    {
        vector<int> rem(m + 1);
        for (int i = 1; i <= m; i++) rem[i] = cap[i];
        vector<char> used(n + 1, 0);
        ll B = 0;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (!used[j] && w[i][j] <= rem[i]) {
                    used[j] = 1;
                    rem[i] -= w[i][j];
                    B += v[i][j];
                }
            }
        }
        if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

        // ---- read & validate participant assignment: exactly n integers in 0..m ----
        vector<int> a(n + 1, 0);
        vector<ll> load(m + 1, 0);
        for (int j = 1; j <= n; j++) {
            int x = ouf.readInt(0, m, "assignment");
            a[j] = x;
            if (x != 0) load[x] += w[x][j];
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

        // capacity feasibility
        for (int i = 1; i <= m; i++) {
            if (load[i] > cap[i])
                quitf(_wa, "station %d over budget: load %lld > cap %d", i, load[i], cap[i]);
        }

        // objective
        ll F = 0;
        for (int j = 1; j <= n; j++)
            if (a[j] != 0) F += v[a[j]][j];

        double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
        quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    }
    return 0;
}
