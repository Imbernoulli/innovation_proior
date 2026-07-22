#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Cliff Amphitheater: Aiming the Echo".
// family: lattice-wave-focus-timing   objective: MAX
//
// A clapper at S pulses at tick 0. Sound travels on the integer grid with the
// OCTILE lattice metric  D(a,b) = 2*max(|dx|,|dy|) + min(|dx|,|dy|).
// An installed reflector R re-emits toward every listener f; the reflected
// wavelet arrives at tick A(f,R)=D(S,R)+D(R,f) with amplitude +g if (A+p) even
// (constructive) else -g (destructive). The direct pulse reaches f at tick
// D(S,f) with +g0. peak(f) = max over ticks of the summed amplitude at that tick.
//
// Objective  F = sum_f peak(f).  Baseline B = M*g0 (do-nothing field, peak=g0
// per listener). Score sc = min(1000, 100*F/max(1,B)); ratio = sc/1000, so
// installing nothing scores 0.1 and any real gain scores higher (cap 1.0).
//
// Feasibility: c in [0,K]; indices in [1,N]; distinct; no trailing tokens.
// -----------------------------------------------------------------------------

static inline ll octile(ll ax, ll ay, ll bx, ll by){
    ll dx = llabs(ax - bx), dy = llabs(ay - by);
    ll mx = max(dx, dy), mn = min(dx, dy);
    return 2*mx + mn;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll G  = inf.readLong();
    ll sx = inf.readLong();
    ll sy = inf.readLong();
    ll g0 = inf.readLong();
    int M = inf.readInt();
    int N = inf.readInt();
    ll K  = inf.readLong();

    vector<ll> fx(M+1), fy(M+1);
    for (int i = 1; i <= M; i++){ fx[i] = inf.readLong(); fy[i] = inf.readLong(); }
    vector<ll> cx(N+1), cy(N+1), cg(N+1), cp(N+1);
    for (int j = 1; j <= N; j++){
        cx[j] = inf.readLong(); cy[j] = inf.readLong();
        cg[j] = inf.readLong(); cp[j] = inf.readLong();
    }

    ll B = (ll)M * g0;
    if (B <= 0) B = 1;

    // ---- read participant's installed set ----
    ll c = ouf.readLong(0, K, "count");
    vector<int> pick;
    vector<char> used(N+1, 0);
    for (ll i = 0; i < c; i++){
        int idx = ouf.readInt(1, N, "index");
        if (used[idx]) quitf(_wa, "candidate %d installed more than once", idx);
        used[idx] = 1;
        pick.push_back(idx);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the install list");

    // ---- compute F = sum_f peak(f) ----
    ll F = 0;
    for (int f = 1; f <= M; f++){
        unordered_map<ll,ll> bins;
        bins.reserve(pick.size()*2 + 4);
        ll directA = octile(sx, sy, fx[f], fy[f]);
        bins[directA] += g0;                       // direct pulse, +g0
        for (int idx : pick){
            ll A = octile(sx, sy, cx[idx], cy[idx]) + octile(cx[idx], cy[idx], fx[f], fy[f]);
            ll sign = (((A + cp[idx]) & 1LL) == 0) ? 1 : -1;
            bins[A] += sign * cg[idx];
        }
        ll best = LLONG_MIN;
        for (auto &kv : bins) best = max(best, kv.second);
        F += best;                                  // peak(f)
    }

    double sc = 100.0 * (double)F / (double)max((ll)1, B);
    if (sc < 0.0) sc = 0.0;
    if (sc > 1000.0) sc = 1000.0;
    quitp(sc / 1000.0, "OK F=%lld B=%lld c=%lld Ratio: %.6f", F, B, c, sc / 1000.0);
    return 0;
}
