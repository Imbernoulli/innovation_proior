#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int T = inf.readInt();
    ll warm = inf.readInt();

    vector<ll> od(T + 1), sp(T + 1);
    vector<int> C(T + 1);
    for (int t = 1; t <= T; t++) od[t] = inf.readInt();
    for (int t = 1; t <= T; t++) sp[t] = inf.readInt();
    for (int t = 1; t <= T; t++) C[t] = inf.readInt();

    vector<int> R(N + 1), d(N + 1);
    for (int j = 1; j <= N; j++) { R[j] = inf.readInt(); d[j] = inf.readInt(); }

    // ---- internal baseline B: each well on-demand contiguously on steps 1..R_j ----
    ll B = 0;
    for (int j = 1; j <= N; j++) {
        B += warm;
        for (int t = 1; t <= R[j]; t++) B += od[t];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant schedule ----
    vector<ll> spotUsed(T + 1, 0);
    ll F = 0;
    for (int j = 1; j <= N; j++) {
        int prev = 0;           // previous step active?
        int progress = 0;       // active steps within [1, d_j]
        for (int t = 1; t <= T; t++) {
            int mode = ouf.readInt(0, 2, "mode");
            int active = (mode != 0);
            if (mode == 1) {
                spotUsed[t]++;
                if (spotUsed[t] > C[t])
                    quitf(_wa, "step %d: spot demand exceeds capacity C=%d", t, C[t]);
                F += sp[t];
            } else if (mode == 2) {
                F += od[t];
            }
            if (active && !prev) F += warm;   // activation warm-up
            if (active && t <= d[j]) progress++;
            prev = active;
        }
        if (progress < R[j])
            quitf(_wa, "well %d: progress %d < required %d by deadline %d",
                  j, progress, R[j], d[j]);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
