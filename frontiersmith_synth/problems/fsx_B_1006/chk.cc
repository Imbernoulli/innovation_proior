#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker/scorer for "Cascade Reservoir Chain: Scheduling Irreversible Releases".
//
// Physics, replayed in order t=1..T, i=1..K (matches statement exactly):
//   pre        = storI_i(t-1) + arrival_i(t)
//   draw       = min(pre, D_i(t));           deficit_i(t) = D_i(t) - draw
//   postdraw   = pre - draw
//   spill_i(t) = max(0, postdraw - Cap_i)    (FORCED, happens before release read)
//   avail_i(t) = postdraw - spill_i(t)       (= min(postdraw, Cap_i))
//   penalty   += spill_i(t)*fw_i + deficit_i(t)*dw_i
//   r_i(t) in [0, avail_i(t)]  <- participant's release, read here
//   storI_i(t) = avail_i(t) - r_i(t)
//   if i<K and t+lag_i<=T: arrival_{i+1}(t+lag_i) += r_i(t) + spill_i(t)
//
// F = participant's total penalty. B = penalty of the fixed "release nothing,
// ever" plan under the identical physics (computed internally, B>0 guaranteed
// since every generated test has positive demand and/or an inflow exceeding
// some capacity). Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int K = inf.readInt();
    int T = inf.readInt();
    vector<ll> Cap(K + 1), S0(K + 1), fw(K + 1), dw(K + 1), lag(K + 1);
    for (int i = 1; i <= K; i++) {
        Cap[i] = inf.readLong();
        S0[i]  = inf.readLong();
        fw[i]  = inf.readLong();
        dw[i]  = inf.readLong();
    }
    for (int i = 1; i < K; i++) lag[i] = inf.readLong();
    vector<ll> I(T + 1);
    for (int t = 1; t <= T; t++) I[t] = inf.readLong();
    vector<vector<ll>> D(K + 1, vector<ll>(T + 1));
    for (int i = 1; i <= K; i++)
        for (int t = 1; t <= T; t++) D[i][t] = inf.readLong();

    // ---- internal baseline B: "release nothing, ever" under identical physics ----
    ll B = 0;
    {
        vector<vector<ll>> arrival(K + 2, vector<ll>(T + 1, 0));
        for (int t = 1; t <= T; t++) arrival[1][t] = I[t];
        vector<ll> storI(K + 1);
        for (int i = 1; i <= K; i++) storI[i] = S0[i];
        for (int t = 1; t <= T; t++) {
            for (int i = 1; i <= K; i++) {
                ll pre = storI[i] + arrival[i][t];
                ll draw = min(pre, D[i][t]);
                ll deficit = D[i][t] - draw;
                ll postdraw = pre - draw;
                ll spill = max(0LL, postdraw - Cap[i]);
                ll avail = postdraw - spill;
                B += spill * fw[i] + deficit * dw[i];
                ll r = 0; // do nothing
                storI[i] = avail - r;
                if (i < K) {
                    ll ft = t + lag[i];
                    if (ft <= T) arrival[i + 1][ft] += r + spill;
                }
            }
        }
        if (B <= 0) B = 1;
    }

    // ---- replay participant's output under identical physics ----
    ll F = 0;
    {
        vector<vector<ll>> arrival2(K + 2, vector<ll>(T + 1, 0));
        for (int t = 1; t <= T; t++) arrival2[1][t] = I[t];
        vector<ll> storI2(K + 1);
        for (int i = 1; i <= K; i++) storI2[i] = S0[i];
        for (int t = 1; t <= T; t++) {
            for (int i = 1; i <= K; i++) {
                ll pre = storI2[i] + arrival2[i][t];
                ll draw = min(pre, D[i][t]);
                ll deficit = D[i][t] - draw;
                ll postdraw = pre - draw;
                ll spill = max(0LL, postdraw - Cap[i]);
                ll avail = postdraw - spill;
                F += spill * fw[i] + deficit * dw[i];
                ll r = ouf.readLong(0, avail, "release");
                storI2[i] = avail - r;
                if (i < K) {
                    ll ft = t + lag[i];
                    if (ft <= T) arrival2[i + 1][ft] += r + spill;
                }
            }
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the T*K releases");
    }

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
