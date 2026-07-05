#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int r = inf.readInt();

    vector<int> row(N + 1), col(N + 1);
    vector<ll>  cost(N + 1);
    vector<int> art(N + 1);
    vector<int> artIdx;            // site indices that are artifacts
    ll B = 0;                      // baseline: a beacon on every artifact
    for (int i = 1; i <= N; i++) {
        row[i]  = inf.readInt();
        col[i]  = inf.readInt();
        cost[i] = inf.readInt();
        art[i]  = inf.readInt();
        if (art[i] == 1) { artIdx.push_back(i); B += cost[i]; }
    }
    if (artIdx.empty() || B <= 0)
        quitf(_fail, "bad instance: artifacts=%d B=%lld", (int)artIdx.size(), B);

    // ---- read & validate participant's beacon set ----
    int K = ouf.readInt(0, N, "K");
    vector<char> chosen(N + 1, 0);
    vector<int>  sel;
    ll F = 0;
    for (int i = 0; i < K; i++) {
        int idx = ouf.readInt(1, N, "beaconIndex");
        if (chosen[idx]) quitf(_wa, "beacon %d installed more than once", idx);
        chosen[idx] = 1;
        sel.push_back(idx);
        F += cost[idx];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- feasibility: every artifact recorded by some chosen beacon ----
    for (int a : artIdx) {
        bool covered = false;
        for (int b : sel) {
            int dr = abs(row[a] - row[b]);
            int dc = abs(col[a] - col[b]);
            if (max(dr, dc) <= r) { covered = true; break; }
        }
        if (!covered)
            quitf(_wa, "artifact at (%d,%d) not recorded by any beacon", row[a], col[a]);
    }

    if (F <= 0) quitf(_wa, "no beacons installed but artifacts exist");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
