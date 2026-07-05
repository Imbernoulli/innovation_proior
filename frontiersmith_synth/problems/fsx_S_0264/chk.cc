#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int T, N;
ll R, G;
vector<ll> f, s;      // 1..T
vector<ll> cap;       // 1..T
vector<int> r, dl;    // 0..N-1
vector<ll> d;         // 0..N-1

// Cost of running the pump at step t given rain/mains sourced there.
static inline ll stepCost(int t, ll rain, ll mains) {
    ll c = 0;
    if (rain + mains > 0) c += f[t];
    c += s[t] * rain;
    c += G * mains;
    return c;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    T = inf.readInt();
    N = inf.readInt();
    R = inf.readLong();
    G = inf.readLong();
    f.assign(T + 1, 0); s.assign(T + 1, 0); cap.assign(T + 1, 0);
    for (int t = 1; t <= T; t++) {
        f[t]   = inf.readLong();
        s[t]   = inf.readLong();
        cap[t] = inf.readLong();
    }
    r.assign(N, 0); dl.assign(N, 0); d.assign(N, 0);
    ll totalDemand = 0;
    for (int i = 0; i < N; i++) {
        r[i]  = inf.readInt();
        dl[i] = inf.readInt();
        d[i]  = inf.readLong();
        totalDemand += d[i];
    }

    // ---- internal baseline B: water each bed at its release step, no storage ----
    // (rain first up to cap_t, then mains for the overflow; L stays 0 throughout).
    vector<ll> Db(T + 1, 0);
    for (int i = 0; i < N; i++) Db[r[i]] += d[i];
    ll B = 0;
    for (int t = 1; t <= T; t++) {
        if (Db[t] <= 0) continue;
        ll rain = min(cap[t], Db[t]);
        ll mains = Db[t] - rain;
        B += stepCost(t, rain, mains);
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline cost B=%lld not positive", B);

    // ---- read & validate participant bed assignment ----
    vector<ll> Demand(T + 1, 0);
    for (int i = 0; i < N; i++) {
        int t = ouf.readInt(1, T, "step");
        if (t < r[i] || t > dl[i])
            quitf(_wa, "bed %d watered at step %d outside window [%d,%d]", i + 1, t, r[i], dl[i]);
        Demand[t] += d[i];
    }

    // ---- read supply plan, simulate the cistern, accumulate cost ----
    ll mainsHi = R + totalDemand;   // generous upper bound on mains at a step
    ll L = 0;
    ll F = 0;
    for (int t = 1; t <= T; t++) {
        ll rain  = ouf.readLong(0, cap[t], "rain");
        ll mains = ouf.readLong(0, mainsHi, "mains");
        ll avail = L + rain + mains;
        if (avail < Demand[t])
            quitf(_wa, "step %d: available %lld < demand %lld (not enough water)", t, avail, Demand[t]);
        L = avail - Demand[t];
        if (L > R)
            quitf(_wa, "step %d: cistern level %lld exceeds capacity R=%lld", t, L, R);
        F += stepCost(t, rain, mains);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (F <= 0) quitf(_wa, "non-positive participant cost F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
