#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int C = inf.readInt();

    vector<int> eu(m), ev(m);
    vector<ll> ew(m);
    ll B = 0; // do-nothing baseline: all depots on one circuit => every link conflicts
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
        B += ew[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate the participant's channel assignment ----
    vector<int> col(n + 1);
    for (int i = 1; i <= n; i++)
        col[i] = ouf.readInt(1, C, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective: total interference (monochromatic edge weight) ----
    ll F = 0;
    for (int i = 0; i < m; i++)
        if (col[eu[i]] == col[ev[i]]) F += ew[i];

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
