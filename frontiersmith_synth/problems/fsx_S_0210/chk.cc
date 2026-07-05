#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int P = inf.readInt();
    int B = inf.readInt();

    vector<ll> cap(P + 1);
    for (int p = 1; p <= P; p++) cap[p] = inf.readInt();

    // w[p][j], v[p][j]
    vector<vector<ll>> w(P + 1, vector<ll>(B + 1));
    vector<vector<ll>> v(P + 1, vector<ll>(B + 1));
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++) {
            w[p][j] = inf.readInt();
            v[p][j] = inf.readInt();
        }

    // ---- internal baseline B0: first-fit (block order, pump order) ----
    vector<ll> rem = cap;
    ll B0 = 0;
    for (int j = 1; j <= B; j++)
        for (int p = 1; p <= P; p++)
            if (rem[p] >= w[p][j]) { rem[p] -= w[p][j]; B0 += v[p][j]; break; }
    if (B0 <= 0) quitf(_fail, "bad instance: baseline B0=%lld", B0);

    // ---- read & validate participant's schedule: B integers in 0..P ----
    vector<ll> load(P + 1, 0);
    ll F = 0;
    for (int j = 1; j <= B; j++) {
        int a = ouf.readInt(0, P, format("a[%d]", j).c_str());
        if (a == 0) continue;
        load[a] += w[a][j];
        if (load[a] > cap[a])
            quitf(_wa, "pump %d over capacity: load %lld > cap %lld", a, load[a], cap[a]);
        F += v[a][j];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B0));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B0, sc / 1000.0);
    return 0;
}
