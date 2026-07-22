#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Additive Interference Spectrum Packing".
//
// Input:  N C K ; alpha theta N0 ; K power levels P_1<...<P_K ;
//         N lines tx ty rx ry w.
// Output: N lines "c k" — c=-1,k=-1 means OFF; else 1<=c<=C, 1<=k<=K means ON
//         on channel c at power level P_k.
//
// gain(a,b) = 1 / D(a,b)^(alpha/2), D = max(1, squared Euclidean distance).
// For ON link i: signal_i = P_{k_i} * gain(tx_i,rx_i);
//   interference_i = N0 + sum_{j ON, j!=i, c_j=c_i} P_{k_j} * gain(tx_j,rx_i);
//   SINR_i = signal_i / interference_i; success iff SINR_i >= theta.
// Objective (MAX): F = sum of w_i over successful i.
//
// Baseline B (checker-computed "no spatial reuse" reference): take the
// min(N,C) highest-value links (ties -> smaller index), give each its own
// channel alone at power P_K. Since every link's own hop has gain 1 by design
// margin and P_K/N0 comfortably exceeds theta, each succeeds; B = sum of
// their values. This is exactly what solutions/trivial.cpp constructs, so it
// reproduces the checker's own baseline (-> ratio 0.1).
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int C = inf.readInt();
    int K = inf.readInt();
    ll alpha = inf.readLong();
    double theta = inf.readDouble();
    ll N0 = inf.readLong();

    vector<ll> P(K + 1);
    for (int j = 1; j <= K; j++) P[j] = inf.readLong();

    vector<ll> tx(N + 1), ty(N + 1), rx(N + 1), ry(N + 1), w(N + 1);
    for (int i = 1; i <= N; i++) {
        tx[i] = inf.readLong();
        ty[i] = inf.readLong();
        rx[i] = inf.readLong();
        ry[i] = inf.readLong();
        w[i] = inf.readLong();
    }

    int m = alpha / 2; // integer exponent applied to D

    auto gain = [&](ll ax, ll ay, ll bx, ll by) -> double {
        ll dx = ax - bx, dy = ay - by;
        ll D = dx * dx + dy * dy;
        if (D < 1) D = 1;
        double Dd = (double)D;
        double denom = 1.0;
        for (int t = 0; t < m; t++) denom *= Dd;
        return 1.0 / denom;
    };

    vector<int> oc(N + 1), ok(N + 1);
    for (int i = 1; i <= N; i++) {
        int c = ouf.readInt(-1, C, "channel");
        int k;
        if (c == -1) {
            k = ouf.readInt(-1, -1, "power_when_off");
        } else {
            k = ouf.readInt(1, K, "power");
        }
        oc[i] = c;
        ok[i] = k;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing data after N lines");

    // Objective F: for each ON link, sum interference from other ON links
    // sharing its channel, then check SINR against theta.
    ll F = 0;
    for (int i = 1; i <= N; i++) {
        if (oc[i] == -1) continue;
        double signal = (double)P[ok[i]] * gain(tx[i], ty[i], rx[i], ry[i]);
        double interference = (double)N0;
        for (int j = 1; j <= N; j++) {
            if (j == i || oc[j] == -1) continue;
            if (oc[j] != oc[i]) continue;
            interference += (double)P[ok[j]] * gain(tx[j], ty[j], rx[i], ry[i]);
        }
        if (!isfinite(signal) || !isfinite(interference) || interference <= 0) continue;
        double sinr = signal / interference;
        if (isfinite(sinr) && sinr >= theta - 1e-9) F += w[i];
    }

    // Baseline B: top min(N,C) values, each alone on its own channel at P_K.
    vector<int> idx(N);
    for (int i = 0; i < N; i++) idx[i] = i + 1;
    stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return w[a] > w[b]; });
    int take = min(N, C);
    ll B = 0;
    for (int t = 0; t < take; t++) B += w[idx[t]];
    if (B < 1) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
