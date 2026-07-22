#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Move the Toll Booths, Shrink the Engines" (retime-and-size-pipeline).
//
// Builds K vertex-disjoint simple-chain "pipelines" covering N stations. testId>=3
// plants ONE dominant "trap" pipeline: nearly all of its original booths are
// clustered on its very first belt, so the pipeline is a single long unbroken
// combinational run under the ORIGINAL booth placement. The shift clock T is
// derived (via a fastest-everywhere simulation) so that this original placement
// is exactly barely feasible with every station at its fastest/priciest variant
// -- guaranteeing the checker's internal baseline B stays feasible. Using the
// cheapest variant everywhere on that same original placement badly violates T
// (trap for a "no retiming, just upsize" greedy). The trap pipeline's total booth
// count is then sized (after T is known) to comfortably exceed the number of
// booths a cheapest-variant chunking would actually need -- so a solver that
// relocates booths can run almost the whole trap pipeline on the cheap variant.
// -----------------------------------------------------------------------------

struct Variant { ll d, a; };

int chunkBreaksNeeded(const vector<ll>& cheapDelay, ll T) {
    int breaks = 0;
    ll run = 0;
    for (size_t i = 0; i < cheapDelay.size(); i++) {
        ll d = cheapDelay[i];
        if (run > 0 && run + d > T) { breaks++; run = d; }
        else run += d;
    }
    return breaks;
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const int Ntab[11]        = {0, 6, 22, 60, 150, 300, 480, 750, 1050, 1350, 1650};
    static const double trapFrac[11] = {0, 0.0, 0.0, 0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.70, 0.72};

    int N = Ntab[testId];
    int trapLen = (int)llround(N * trapFrac[testId]);
    if (trapLen > 0 && trapLen < 4) trapLen = 0; // too small to be a meaningful trap
    if (trapLen > 0 && (N - trapLen) == 1) trapLen -= 1; // avoid a stray 1-station filler

    // ---- pipeline length layout ----
    vector<int> lens;
    if (trapLen > 0) lens.push_back(trapLen);
    int remaining = N - (trapLen > 0 ? trapLen : 0);
    while (remaining > 0) {
        int Lp;
        if (remaining <= 9) Lp = remaining;
        else Lp = (int)rnd.next(2, 8);
        if (remaining - Lp == 1) { if (Lp > 2) Lp -= 1; else Lp += 1; }
        if (Lp > remaining) Lp = remaining;
        lens.push_back(Lp);
        remaining -= Lp;
    }
    int K = (int)lens.size();

    // ---- per-station engine menus ----
    vector<int> Kv(N + 1);
    vector<array<Variant,3>> menu(N + 1);
    for (int v = 1; v <= N; v++) {
        int kv = (rnd.next(0, 99) < 50) ? 2 : 3;
        Kv[v] = kv;
        ll slowD = rnd.next(15, 40), slowA = rnd.next(1, 6);
        ll fastD = rnd.next(2, 10);
        ll fastA = slowA + rnd.next(6, 16);
        menu[v][0] = {slowD, slowA};
        if (kv == 3) {
            ll midD = rnd.next(fastD + 1, slowD - 1);
            ll midA = rnd.next(slowA + 1, fastA - 1);
            menu[v][1] = {midD, midA};
            menu[v][2] = {fastD, fastA};
        } else {
            menu[v][1] = {fastD, fastA};
        }
    }

    // ---- per-pipeline belts: original booth count w, relocation cost c ----
    // pipeline p (0-indexed in `lens`) owns belts local 1..len-1
    vector<vector<ll>> w(K), c(K);
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        w[p].assign(L - 1, 0);
        c[p].assign(L - 1, 0);
        bool isTrap = (trapLen > 0 && p == 0);
        for (int i = 0; i < L - 1; i++) {
            c[p][i] = rnd.next(1, 20);
            if (isTrap) {
                w[p][i] = (i == 0) ? 1 : 0; // placeholder pattern; magnitude fixed after T is known
            } else {
                w[p][i] = (rnd.next(1, 100) <= 55) ? 1 : 0;
            }
        }
    }

    // station offsets per pipeline
    vector<int> off(K);
    { int cur = 0; for (int p = 0; p < K; p++) { off[p] = cur; cur += lens[p]; } }

    // ---- compute T: fastest-everywhere arrival simulation over the ORIGINAL pattern ----
    ll globalMaxArrival = 0;
    for (int p = 0; p < K; p++) {
        int L = lens[p];
        ll run = 0;
        for (int i = 0; i < L; i++) {
            int v = off[p] + i + 1;
            ll fd = menu[v][Kv[v] - 1].d;
            bool freshRun = (i == 0) || (w[p][i - 1] >= 1);
            run = freshRun ? fd : (run + fd);
            bool checkpoint = (i == L - 1) || (i < L - 1 && w[p][i] >= 1);
            if (checkpoint) globalMaxArrival = max(globalMaxArrival, run);
        }
    }
    ll T = (ll)ceil(globalMaxArrival * 1.08) + 3;

    // safety clamp: no single station's cheapest delay may exceed T (keeps the
    // trap-pipeline chunking guarantee well-defined for every test size)
    for (int v = 1; v <= N; v++) if (menu[v][0].d > T) menu[v][0].d = max(1LL, T - 1);

    // ---- size the trap pipeline's total booth count from T ----
    if (trapLen > 0) {
        int L = trapLen;
        vector<ll> cheap(L);
        for (int i = 0; i < L; i++) cheap[i] = menu[off[0] + i + 1][0].d;
        int need = chunkBreaksNeeded(cheap, T);
        ll spare = rnd.next(2, 5);
        ll Wtot = (ll)need + spare;
        if (Wtot < 1) Wtot = 1;
        w[0][0] = Wtot; // all original booths clustered on the pipeline's first belt
    }

    // ---- emit ----
    printf("%d %d %lld\n", N, K, T);
    for (int p = 0; p < K; p++) printf("%d%c", lens[p], p + 1 == K ? '\n' : ' ');
    for (int v = 1; v <= N; v++) {
        printf("%d", Kv[v]);
        for (int i = 0; i < Kv[v]; i++) printf(" %lld %lld", menu[v][i].d, menu[v][i].a);
        printf("\n");
    }
    for (int p = 0; p < K; p++)
        for (int i = 0; i < lens[p] - 1; i++)
            printf("%lld %lld\n", w[p][i], c[p][i]);

    return 0;
}
