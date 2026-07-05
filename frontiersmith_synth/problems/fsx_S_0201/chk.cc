#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int C = inf.readInt();

    vector<int> eu(M), ev(M);
    vector<ll> ew(M);
    ll B = 0; // baseline: all stations on one channel -> every pair is co-channel
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
        B += ew[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's channel assignment ----
    vector<int> col(N + 1);
    for (int i = 1; i <= N; i++) col[i] = ouf.readInt(1, C, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective: total interference cost ----
    ll F = 0;
    for (int i = 0; i < M; i++) {
        int d = abs(col[eu[i]] - col[ev[i]]);
        if (d == 0)      F += ew[i];
        else if (d == 1) F += ew[i] / 2;
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
