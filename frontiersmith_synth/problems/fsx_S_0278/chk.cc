#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M;
vector<ll> dx, dy;               // demand coords
vector<ll> sx, sy, sr, sc;       // sensor coords, radius, cost

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    dx.resize(N); dy.resize(N);
    for (int i = 0; i < N; i++) { dx[i] = inf.readLong(); dy[i] = inf.readLong(); }
    sx.resize(M); sy.resize(M); sr.resize(M); sc.resize(M);
    ll B = 0; // baseline = cost of switching EVERY sensor on (the trivial feasible net)
    for (int j = 0; j < M; j++) {
        sx[j] = inf.readLong(); sy[j] = inf.readLong();
        sr[j] = inf.readLong(); sc[j] = inf.readLong();
        B += sc[j];
    }
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's activated-sensor set ----
    int K = ouf.readInt(0, M, "K");
    vector<char> used(M, 0);
    vector<char> covered(N, 0);
    ll F = 0;
    for (int a = 0; a < K; a++) {
        int idx = ouf.readInt(1, M, "sensorIndex"); // 1-based; rejects out-of-range/garbage/nan
        int j = idx - 1;
        if (used[j]) quitf(_wa, "sensor %d activated more than once", idx);
        used[j] = 1;
        F += sc[j];
        ll r2 = sr[j] * sr[j];
        for (int i = 0; i < N; i++) {
            if (covered[i]) continue;
            ll ddx = sx[j] - dx[i], ddy = sy[j] - dy[i];
            if (ddx * ddx + ddy * ddy <= r2) covered[i] = 1;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int i = 0; i < N; i++)
        if (!covered[i]) quitf(_wa, "demand station %d not covered", i + 1);

    if (F <= 0) quitf(_wa, "empty / zero-cost net cannot cover %d stations", N);

    double sc_ = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc_ / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc_ / 1000.0);
    return 0;
}
