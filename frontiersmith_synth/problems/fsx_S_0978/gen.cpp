#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Cross-Dock Door Draw and Pallet Relay.
// testId is a difficulty/structure ladder: tiny sanity at 1, growing to the full
// constraint envelope (T_in=T_out=30, M=400) at 10. Structure varies deliberately:
// dense/near-complete bipartite flow tests are "filler" tests where no single
// arrangement can perfectly avoid every overlap; sparse "needle"/"matching"/
// "clustered" tests are deliberate TRAPS where a solver that never touches the door
// draw (only batches pallets on the input-order block layout) gets badly punished,
// because a handful of huge flows are placed at maximum identity-order distance (or
// share near-identical identity-order distance) and so overlap heavily on the same
// central segments -- while a door draw that puts matched inbound/outbound trucks
// adjacent removes almost all of that overlap.
//
// `cap` (segment throughput) is DERIVED from each test's own total pallet volume (via
// capMult) rather than hard-coded, so the nonlinear congestion regime scales
// consistently with how much traffic actually exists on that test.

struct Pair { int i, j, f; };

static void emit(int Tin, int Tout, int K, double capMult, map<pair<int,int>,long long>& flows) {
    // clip flows into [1,200] and drop non-positive
    vector<Pair> ps;
    long long total = 0;
    for (auto& kv : flows) {
        long long f = kv.second;
        if (f <= 0) continue;
        if (f > 200) f = 200;
        ps.push_back({kv.first.first, kv.first.second, (int)f});
        total += f;
    }
    if ((int)ps.size() > 400) ps.resize(400);
    if (ps.empty()) { ps.push_back({1, 1, 1}); total = 1; }
    long long cap = max(4LL, (long long)llround(capMult * (double)total));
    if (cap > 600000) cap = 600000;
    printf("%d %d %d %d %lld\n", Tin, Tout, (int)ps.size(), K, cap);
    for (auto& p : ps) printf("%d %d %d\n", p.i, p.j, p.f);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    map<pair<int,int>,long long> flows;

    if (idx == 1) {
        // tiny sanity, matches the statement's worked example scale: COMPLETE
        // bipartite (every inbound owes every outbound), so no arrangement can avoid
        // all overlap -- keeps this smallest test from trivially saturating.
        int Tin = 2, Tout = 2, K = 3;
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                flows[{i,j}] = rnd.next(3, 6);
        emit(Tin, Tout, K, 8.0, flows);
        return 0;
    }

    if (idx == 2) {
        // small, COMPLETE bipartite flow -- dense enough that no single 1-D
        // arrangement can avoid every overlap, so even a strong reordering has a
        // real floor on how much it can win (unlike a sparse graph, which a good
        // linear arrangement can lay out almost overlap-free).
        int Tin = 3, Tout = 3, K = 4;
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                flows[{i,j}] = rnd.next(5, 40);
        emit(Tin, Tout, K, 50.0, flows);
        return 0;
    }

    if (idx == 3) {
        // medium, COMPLETE bipartite flow -- no single dominant trap, but no
        // perfectly overlap-free arrangement exists either.
        int Tin = 4, Tout = 4, K = 5;
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                flows[{i,j}] = rnd.next(5, 35);
        emit(Tin, Tout, K, 50.0, flows);
        return 0;
    }

    if (idx == 4) {
        // TRAP: near-perfect matching inbound k <-> outbound k, all large flow, on a
        // sparse graph. Under the identity block draw every pair spans distance
        // ~Tin (roughly the same span), so ALL of them overlap on the shared middle
        // segments -- a door draw that interleaves k with k removes almost all of it.
        int Tin = 8, Tout = 8, K = 6;
        for (int k = 1; k <= Tin; k++) flows[{k, k}] = rnd.next(90, 150);
        for (int k = 0; k < 5; k++) {
            int i = rnd.next(1, Tin), j = rnd.next(1, Tout);
            flows[{i,j}] += rnd.next(1, 6);
        }
        emit(Tin, Tout, K, 6.0, flows);
        return 0;
    }

    if (idx == 5) {
        // TRAP: a "needle" pair (inbound 1 <-> outbound T_out) planted at the MAXIMUM
        // possible identity-order distance (it crosses the entire aisle under the
        // identity block draw), plus a second large flow (inbound T_in <-> outbound
        // 1) that sits at the identity-order boundary itself, and small noise
        // elsewhere. A door draw that reorders trucks so the needle's two ends sit
        // adjacent removes almost the whole aisle-spanning crossing.
        int Tin = 8, Tout = 8, K = 6;
        flows[{1, Tout}] = 180;
        flows[{Tin, 1}] = 180;
        for (int k = 0; k < 8; k++) {
            int i = rnd.next(1, Tin), j = rnd.next(1, Tout);
            flows[{i,j}] += rnd.next(1, 8);
        }
        emit(Tin, Tout, K, 6.0, flows);
        return 0;
    }

    if (idx == 6) {
        // larger, COMPLETE bipartite filler.
        int Tin = 10, Tout = 10, K = 6;
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                flows[{i,j}] = rnd.next(3, 30);
        emit(Tin, Tout, K, 80.0, flows);
        return 0;
    }

    if (idx == 7) {
        // TRAP: clustered block-diagonal matching -- several disjoint matched
        // clusters, each a small perfect-matching-like group with heavy internal
        // flow. Identity order interleaves inbound/outbound blocks so clusters still
        // overlap heavily in the middle; a good arrangement keeps each cluster's
        // trucks contiguous and adjacent.
        int Tin = 14, Tout = 14, K = 6;
        int nclusters = 4;
        for (int c = 0; c < nclusters; c++) {
            vector<int> ins, outs;
            for (int k = 0; k < Tin / nclusters; k++) ins.push_back(c * (Tin/nclusters) + k + 1);
            for (int k = 0; k < Tout / nclusters; k++) outs.push_back(c * (Tout/nclusters) + k + 1);
            for (int a : ins)
                for (int b : outs)
                    if (rnd.next(0, 99) < 55) flows[{a,b}] += rnd.next(20, 70);
        }
        for (int k = 0; k < 12; k++) {
            int i = rnd.next(1, Tin), j = rnd.next(1, Tout);
            flows[{i,j}] += rnd.next(1, 5);
        }
        emit(Tin, Tout, K, 8.0, flows);
        return 0;
    }

    if (idx == 8) {
        // large, COMPLETE bipartite filler (M = Tin*Tout = 400, at the M cap) --
        // stresses the sheer number of pairs/trips while the density itself bounds
        // how much any single arrangement can win.
        int Tin = 20, Tout = 20, K = 7;
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                flows[{i,j}] = rnd.next(2, 25);
        emit(Tin, Tout, K, 80.0, flows);
        return 0;
    }

    if (idx == 9) {
        // large TRAP: near-perfect matching at scale.
        int Tin = 20, Tout = 20, K = 8;
        for (int k = 1; k <= Tin; k++) {
            int j = min(Tout, k + rnd.next(-1, 1));
            if (j < 1) j = 1;
            flows[{k, j}] += rnd.next(80, 150);
        }
        for (int k = 0; k < 25; k++) {
            int i = rnd.next(1, Tin), j = rnd.next(1, Tout);
            flows[{i,j}] += rnd.next(1, 6);
        }
        emit(Tin, Tout, K, 10.0, flows);
        return 0;
    }

    // idx == 10: fill the full envelope -- T_in=T_out=30 (D=60), M near 400,
    // mixed needle + matching + noise so both mechanisms stay decisive at scale.
    {
        int Tin = 30, Tout = 30, K = 8;
        flows[{1, Tout}] += 200;
        flows[{Tin, 1}] += 200;
        for (int k = 1; k <= Tin; k++) if (rnd.next(0, 99) < 70) flows[{k, k}] += rnd.next(40, 120);
        for (int i = 1; i <= Tin; i++)
            for (int j = 1; j <= Tout; j++)
                if (rnd.next(0, 99) < 15) flows[{i,j}] += rnd.next(1, 15);
        emit(Tin, Tout, K, 20.0, flows);
        return 0;
    }
}
