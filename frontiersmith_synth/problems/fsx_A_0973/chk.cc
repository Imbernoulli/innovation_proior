#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Phased-Array Drift-Anchored Calibration Sweep".
//
// Input:  N T ; then N lines x_i y_i d_i  (d_0 = 0, the master reference).
// Output: T lines "u v" -- the measured pair at each time step t=1..T.
//
// Replays the exact state machine from the statement:
//   connected[i], err[i] (long long), last[i] (long long), connected[0]=true,
//   err[0]=0, last[0]=0.
//   neither connected            -> no-op
//   exactly one connected (a,b)  -> err[b] = err[a] + d[a]*(t-last[a])
//                                            + d[b]*t + w(u,v);
//                                    last[a]=last[b]=t; connected[b]=true
//   both connected (u,v)         -> cand_u = err[v]+d[v]*(t-last[v])
//                                             +d[u]*(t-last[u])+w(u,v)
//                                    (symmetric cand_v); err[*]=min(err[*],cand_*);
//                                    last[u]=last[v]=t
//
// Feasibility: T distinct-endpoint pairs in range; every element 1..N-1 ends
// connected. Any violation -> quitf(_wa, ...) -> score 0.
//
// Objective (MIN): F = sum_{i=1}^{N-1} err_i.
//
// Baseline B (checker-computed): the naive GROUPED CHAIN through raw index
// order -- elements are cabled together in fixed-size runs of GROUP_SIZE:
// within a run, each new element measures against the PREVIOUS element
// (1,2),(2,3),...; every GROUP_SIZE-th element instead resets to the
// reference (0, k). Concretely, for i=1..N-1: parent(i) = 0 if
// (i-1) % GROUP_SIZE == 0, else i-1. This ignores geometry, drift and
// timing, and because err propagates additively along the measurement path,
// each run still inherits its own earlier hops' error at later nodes in the
// SAME run -- but resetting periodically keeps that inheritance bounded
// instead of compounding across the whole field. This is exactly what
// solutions/trivial.cpp emits.
//
// Score (min): sc = min(1000, 100*B/max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static inline ll wdist(ll x1, ll y1, ll x2, ll y2) {
    return llabs(x1 - x2) + llabs(y1 - y2);
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    ll T = inf.readLong();
    vector<ll> X(N), Y(N), D(N);
    for (int i = 0; i < N; i++) {
        X[i] = inf.readLong();
        Y[i] = inf.readLong();
        D[i] = inf.readLong();
    }

    // ---- internal baseline B: naive grouped chain, GROUP_SIZE-run resets ----
    const int GROUP_SIZE = 6;
    ll B = 0;
    {
        vector<ll> berr(N, 0), blast(N, 0);
        for (int i = 1; i < N; i++) {
            int p = ((i - 1) % GROUP_SIZE == 0) ? 0 : (i - 1);
            ll deltaP = D[p] * ((ll)i - blast[p]);
            ll deltaI = D[i] * (ll)i;
            ll errCur = berr[p] + deltaP + deltaI + wdist(X[p], Y[p], X[i], Y[i]);
            berr[i] = errCur;
            blast[p] = i; blast[i] = i;
            B += errCur;
        }
    }
    if (B <= 0) B = 1;

    // ---- replay participant schedule ----
    vector<char> connected(N, 0);
    vector<ll> err(N, 0), last(N, 0);
    connected[0] = 1; err[0] = 0; last[0] = 0;

    for (ll t = 1; t <= T; t++) {
        int u = ouf.readInt(0, N - 1, "u");
        int v = ouf.readInt(0, N - 1, "v");
        if (u == v) quitf(_wa, "step %lld: u == v (%d)", t, u);

        bool cu = connected[u], cv = connected[v];
        ll w = wdist(X[u], Y[u], X[v], Y[v]);

        if (!cu && !cv) {
            // wasted step: no state change
            continue;
        } else if (cu != cv) {
            int a = cu ? u : v;
            int b = cu ? v : u;
            ll deltaA = D[a] * (t - last[a]);
            ll deltaB = D[b] * t;
            err[b] = err[a] + deltaA + deltaB + w;
            connected[b] = 1;
            last[a] = t;
            last[b] = t;
        } else {
            ll deltaU = D[u] * (t - last[u]);
            ll deltaV = D[v] * (t - last[v]);
            ll candU = err[v] + deltaV + deltaU + w;
            ll candV = err[u] + deltaU + deltaV + w;
            err[u] = min(err[u], candU);
            err[v] = min(err[v], candV);
            last[u] = t;
            last[v] = t;
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %lld steps", T);

    for (int i = 1; i < N; i++) {
        if (!connected[i]) quitf(_wa, "element %d never became connected to the reference", i);
    }

    ll F = 0;
    for (int i = 1; i < N; i++) F += err[i];
    if (F < 0) quitf(_wa, "internal error: negative F");  // arithmetic sanity guard

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
