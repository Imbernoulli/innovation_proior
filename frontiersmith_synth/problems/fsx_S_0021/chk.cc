#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, K;
vector<int> eu, ev;
vector<ll> ep, eq;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    K = inf.readInt();

    eu.resize(m); ev.resize(m); ep.resize(m); eq.resize(m);
    ll B = 0; // baseline: everyone on channel 1 -> every pair is co-channel
    for (int i = 0; i < m; i++) {
        eu[i] = inf.readInt();
        ev[i] = inf.readInt();
        ep[i] = inf.readInt();
        eq[i] = inf.readInt();
        B += ep[i];
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate the participant's channel assignment ----
    vector<int> c(n + 1);
    for (int i = 1; i <= n; i++)
        c[i] = ouf.readInt(1, K, "channel");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- objective F ----
    ll F = 0;
    for (int i = 0; i < m; i++) {
        int a = c[eu[i]], b = c[ev[i]];
        if (a == b) F += ep[i];
        else if (abs(a - b) == 1) F += eq[i];
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
