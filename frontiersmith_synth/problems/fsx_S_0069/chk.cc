#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int K = inf.readInt();
    int M = inf.readInt();

    vector<int> eu(M), ev(M), es(M), ep(M);
    ll B = 0; // baseline: every radio on channel 1 -> full p*s per pair
    for (int i = 0; i < M; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        es[i] = inf.readInt();
        ep[i] = inf.readInt();
        B += (ll)ep[i] * (ll)es[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant's channel assignment ----
    vector<int> c(N + 1);
    for (int i = 1; i <= N; i++)
        c[i] = ouf.readInt(1, K, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective ----
    ll F = 0;
    for (int i = 0; i < M; i++) {
        int diff = abs(c[eu[i]] - c[ev[i]]);
        int shortfall = es[i] - diff;
        if (shortfall > 0) F += (ll)ep[i] * (ll)shortfall;
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
