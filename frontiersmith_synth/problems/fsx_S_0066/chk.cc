#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int P = inf.readInt();
    int B = inf.readInt();
    vector<long long> S(P + 1);
    for (int j = 1; j <= P; j++) S[j] = inf.readLong();

    // reachable[i] = list of (patch, appetite); first entry = lowest-indexed patch.
    vector<vector<pair<int,int>>> reach(B + 1);
    for (int i = 1; i <= B; i++) {
        int m = inf.readInt();
        reach[i].reserve(m);
        for (int k = 0; k < m; k++) {
            int p = inf.readInt();
            int a = inf.readInt();
            reach[i].push_back({p, a});
        }
    }

    // ---- read participant dispatch, validate feasibility ----
    vector<long long> load(P + 1, 0);
    for (int i = 1; i <= B; i++) {
        int d = ouf.readInt(0, P, format("d[%d]", i));
        if (d == 0) continue;
        // patch d must be reachable by forager i
        int app = -1;
        for (auto& pr : reach[i])
            if (pr.first == d) { app = pr.second; break; }
        if (app < 0)
            quitf(_wa, "forager %d dispatched to unreachable patch %d", i, d);
        load[d] += app;
    }
    if (!ouf.seekEof())
        quitf(_wa, "trailing tokens after %d dispatches", B);

    long long F = 0;
    for (int j = 1; j <= P; j++) F += min(S[j], load[j]);

    // ---- internal baseline B*: every forager to its lowest-indexed reachable patch ----
    vector<long long> loadB(P + 1, 0);
    for (int i = 1; i <= B; i++) {
        auto& pr = reach[i][0]; // lowest index (input is sorted ascending)
        loadB[pr.first] += pr.second;
    }
    long long Bstar = 0;
    for (int j = 1; j <= P; j++) Bstar += min(S[j], loadB[j]);
    if (Bstar < 1) Bstar = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, Bstar));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, Bstar, sc / 1000.0);
    return 0;
}
