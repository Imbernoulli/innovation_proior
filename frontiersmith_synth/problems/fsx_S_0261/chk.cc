#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int K = inf.readInt();

    vector<int> p(n + 1), s(n + 1);
    ll B = 0;                        // do-nothing baseline: everyone on channel 1
    for (int i = 1; i <= n; i++) {
        p[i] = inf.readInt();
        s[i] = inf.readInt();
        B += (ll)s[i] * abs(1 - p[i]);      // retuning stress at channel 1
    }

    vector<int> eu(m), ev(m), ew(m), eg(m);
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ew[i] = inf.readInt();
        eg[i] = inf.readInt();
        B += (ll)ew[i] * eg[i];             // diff = 0 on every pair at channel 1
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's channel allocation ----
    vector<int> c(n + 1);
    for (int i = 1; i <= n; i++)
        c[i] = ouf.readInt(1, K, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective: residual cross-talk + retuning stress ----
    ll F = 0;
    for (int i = 1; i <= n; i++)
        F += (ll)s[i] * abs(c[i] - p[i]);
    for (int i = 0; i < m; i++) {
        int diff = abs(c[eu[i]] - c[ev[i]]);
        int pen = eg[i] - diff;
        if (pen < 0) pen = 0;
        F += (ll)ew[i] * pen;
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
