#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Robotic Telescope Night Plan".
//
// Input:  N P k K R Tb SCALE Dmax ; N lines (p_i v_i) ; then Dmax+1 ints acc[0..Dmax].
// Output (participant): first token A = number of actions; then A actions, each
//    "1 i"  observe target i (1..N, each at most once), or
//    "0"    recalibrate (reset drift to 0).
//
// Simulation (start coordinate 0, drift 0, time 0):
//   observe i: dist=|cur-p_i|; time+=dist; drift+=k*dist; cur=p_i;
//              yield += v_i * acc[min(drift,Dmax)].
//   recalibrate: time+=R; resets+=1; drift=0.
// Feasibility: resets<=K, time<=Tb, each target observed at most once.
// Objective (MAX): F = total yield.
//
// Internal baseline B (checker-built = the reference "monotone sweep, no
// recalibration"): visit targets in coordinate order from 0, never recalibrate,
// sum the yields. This is exactly the shortest-tour construction the trivial
// reference reproduces (-> ratio 0.1). B>0 since acc>=1, v>=1.
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N   = inf.readInt();
    ll  P   = inf.readLong();
    ll  k   = inf.readLong();
    int K   = inf.readInt();
    ll  R   = inf.readLong();
    ll  Tb  = inf.readLong();
    ll  SCALE = inf.readLong();
    ll  Dmax  = inf.readLong();
    (void)P; (void)SCALE;

    vector<ll> px(N + 1), pv(N + 1);
    for (int i = 1; i <= N; i++){ px[i] = inf.readLong(); pv[i] = inf.readLong(); }
    vector<ll> acc(Dmax + 1);
    for (ll d = 0; d <= Dmax; d++) acc[d] = inf.readLong();

    auto accAt = [&](ll drift) -> ll {
        if (drift < 0) drift = 0;
        if (drift > Dmax) drift = Dmax;
        return acc[drift];
    };

    // ---- internal baseline B: monotone sweep from 0, no recalibration ----
    vector<int> order(N);
    for (int i = 0; i < N; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (px[a] != px[b]) return px[a] < px[b];
        return a < b;
    });
    ll B = 0;
    { ll cur = 0, drift = 0;
      for (int id : order){ ll dist = llabs(cur - px[id]); drift += k * dist; cur = px[id];
                            B += pv[id] * accAt(drift); } }
    if (B <= 0) B = 1;

    // ---- replay participant's plan ----
    ll A = ouf.readLong(0LL, (ll)N + (ll)K + 5, "A");
    vector<char> seen(N + 1, 0);
    ll cur = 0, drift = 0, timeUsed = 0, resets = 0, F = 0;
    for (ll a = 0; a < A; a++){
        int typ = ouf.readInt(0, 1, "action_type");
        if (typ == 0){
            resets++;
            if (resets > K) quitf(_wa, "used %lld recalibrations > K=%d", resets, K);
            timeUsed += R;
            drift = 0;
        } else {
            int i = ouf.readInt(1, N, "target");
            if (seen[i]) quitf(_wa, "target %d observed more than once", i);
            seen[i] = 1;
            ll dist = llabs(cur - px[i]);
            timeUsed += dist;
            drift += k * dist;
            cur = px[i];
            F += pv[i] * accAt(drift);
        }
        if (timeUsed > Tb) quitf(_wa, "night time budget exceeded: %lld > Tb=%lld", timeUsed, Tb);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the plan");
    if (timeUsed > Tb) quitf(_wa, "night time budget exceeded: %lld > Tb=%lld", timeUsed, Tb);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld resets=%lld/%d time=%lld/%lld Ratio: %.6f",
          F, B, resets, K, timeUsed, Tb, sc / 1000.0);
    return 0;
}
