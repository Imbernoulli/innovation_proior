#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Shelters Sited for the Worst Flood-Day Traffic"  (generator)
// family: bottleneck-aware-shelter-siting
//
// City road graph = D district gateways (population lives here) + Hn hub nodes
// (elevated ring backbone) + 2D candidate shelter-site nodes:
//   - D "near-fragile" (NF) sites: each attached to its OWN district gateway by a
//     single low-lying road link (looks perfect: distance 1, decent capacity).
//   - D "far-robust" (FR) sites: each attached to a hub by a link off the elevated
//     ring backbone (distance 2 via the hub, smaller individual intake capacity).
//   - Every district also has exactly one "surviving bridge" edge gateway->hub
//     (never blocked in any scenario -- the seed's "single surviving bridge").
//
// K published flood-day blockage scenarios are listed; for testId>=7 ("trap"
// tests) most scenarios sever EVERY near-fragile link at once (flood takes out
// all the low-lying shortcut roads simultaneously) while the elevated ring +
// bridges survive. A distance/coverage greedy always prefers NF sites (they are
// literally the closest possible site to their district, beating every FR site
// by hop-distance) and gets crushed on those scenarios. The correct siting
// reasons about the max-flow (= min surviving cut) from population to shelters
// in EVERY scenario and picks a mix that keeps every scenario's cut healthy.
//
// Candidate presentation order is a random permutation per type; the checker's
// own "first S candidates" baseline mixes in a bounded number of near-fragile
// picks (more on trap-heavy tests, since a proximity-driven pick would plausibly
// include them there too) but is never all-near-fragile -- naive, not adversarial.
//
// Input format:
//   N M C S K
//   p_1 .. p_N                      (population at graph node i, 1-indexed)
//   M lines: u v cap                (undirected road edge, 1<=u,v<=N)
//   C lines: node cap               (candidate i's graph node + shelter intake cap)
//   K lines: b e_1 .. e_b           (blocked edge indices, 0-indexed into the M list)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]); // 1..10

    int D = t + 1;                          // districts: 2..11
    int Hn = max(1, D / 3);                 // hub nodes: 1..3
    int K = 4 + t;                          // published blockage scenarios: 5..14
    bool trapHeavy = (t >= 7);              // testIds 7,8,9,10 are trap-heavy (>=3 of 10)

    int gateway0 = 1;                 // nodes 1..D
    int hub0 = D + 1;                 // nodes D+1..D+Hn
    int nf0 = D + Hn + 1;             // nodes nf0..nf0+D-1
    int fr0 = D + Hn + D + 1;         // nodes fr0..fr0+D-1
    int N = D + Hn + D + D;           // total graph nodes

    auto gateway = [&](int d){ return gateway0 + d; };      // d in 0..D-1
    auto hub     = [&](int h){ return hub0 + h; };           // h in 0..Hn-1
    auto nfNode  = [&](int d){ return nf0 + d; };
    auto frNode  = [&](int d){ return fr0 + d; };

    // ---- populations (only gateway nodes carry population) ----
    vector<ll> pop(N + 1, 0);
    for (int d = 0; d < D; d++) pop[gateway(d)] = rnd.next(50, 150) + 8 * t;

    // ---- edges ----
    struct Edge { int u, v; ll cap; };
    vector<Edge> edges;

    vector<int> bridgeIdx(D), nfIdx(D), frIdx(D);
    for (int d = 0; d < D; d++) {
        int h = d % Hn;
        ll cap = rnd.next(15, 30) + 2 * t;
        bridgeIdx[d] = (int)edges.size();
        edges.push_back({gateway(d), hub(h), cap});
    }
    for (int d = 0; d < D; d++) {
        ll cap = rnd.next(45, 65) + 2 * t;
        nfIdx[d] = (int)edges.size();
        edges.push_back({gateway(d), nfNode(d), cap});
    }
    for (int d = 0; d < D; d++) {
        int h = (d + 1) % Hn;
        ll cap = rnd.next(18, 30) + 1 * t;
        frIdx[d] = (int)edges.size();
        edges.push_back({hub(h), frNode(d), cap});
    }
    vector<int> ringIdx;
    if (Hn >= 2) {
        for (int h = 0; h < Hn; h++) {
            ll cap = rnd.next(50, 90) + 3 * t;
            ringIdx.push_back((int)edges.size());
            edges.push_back({hub(h), hub((h + 1) % Hn), cap});
        }
    }
    int Mblockable = (int)edges.size();

    // "Backroad" edges: NF_d also has a slow, low-capacity emergency link straight
    // to its hub. This is NEVER a scenario's blocked target (flood never severs
    // the elevated emergency line) so a near-fragile pick is degraded, not
    // annihilated, when its primary link is cut -- the trap punishes proximity
    // reasoning without making an all-fragile pick literally worth zero.
    for (int d = 0; d < D; d++) {
        int h = d % Hn;
        ll cap = rnd.next(3, 7) + t / 4;
        edges.push_back({hub(h), nfNode(d), cap});
    }
    int M = (int)edges.size();

    // ---- candidate site capacities ----
    vector<ll> nfSiteCap(D), frSiteCap(D);
    for (int d = 0; d < D; d++) nfSiteCap[d] = rnd.next(110, 170) + 4 * t;
    for (int d = 0; d < D; d++) frSiteCap[d] = rnd.next(12, 22) + t / 2;

    int C = 2 * D;
    int S = max(2, (D + 1) / 2);
    if (S >= C) S = C - 1;

    // ---- candidate presentation order: mostly far-robust, a few near-fragile ----
    // This is what makes the checker's own "first S in presentation order" baseline
    // a plausibly naive (never sees capacities/blockages, just lists sites) but
    // FR-majority pick -- never catastrophic (always trap-safe, since it barely
    // touches near-fragile sites) and never a strong reference either (a solver
    // that actually reasons about the flow can do much better in both regimes).
    vector<int> nfPerm(D), frPerm(D);
    for (int d = 0; d < D; d++) { nfPerm[d] = d; frPerm[d] = d; }
    for (int d = D - 1; d > 0; d--) swap(nfPerm[d], nfPerm[rnd.next(d + 1)]);
    for (int d = D - 1; d > 0; d--) swap(frPerm[d], frPerm[rnd.next(d + 1)]);

    // How many of the first S presented slots are near-fragile. On trap-heavy
    // tests the naive baseline is allowed to lean on more near-fragile sites (so
    // it is genuinely exposed on the published flood scenarios, same as any
    // proximity-driven pick would be); on calm tests it leans mostly far-robust
    // (capacity-limited, so it is a plausible but unremarkable pick there too).
    int nfWanted = trapHeavy ? (S - 1) : 1;
    nfWanted = max(1, min(nfWanted, S - 1));
    vector<int> candNode(C);
    vector<ll> candCap(C);
    {
        int ni = 0, fi = 0;
        for (int pos = 0; pos < S; pos++) {
            bool wantNf = (pos < nfWanted);
            if (wantNf && ni < D) {
                candNode[pos] = nfNode(nfPerm[ni]); candCap[pos] = nfSiteCap[nfPerm[ni]]; ni++;
            } else if (fi < D) {
                candNode[pos] = frNode(frPerm[fi]); candCap[pos] = frSiteCap[frPerm[fi]]; fi++;
            } else {
                candNode[pos] = nfNode(nfPerm[ni]); candCap[pos] = nfSiteCap[nfPerm[ni]]; ni++;
            }
        }
        int pos = S;
        while (ni < D) { candNode[pos] = nfNode(nfPerm[ni]); candCap[pos] = nfSiteCap[nfPerm[ni]]; ni++; pos++; }
        while (fi < D) { candNode[pos] = frNode(frPerm[fi]); candCap[pos] = frSiteCap[frPerm[fi]]; fi++; pos++; }
    }

    // ---- scenarios ----
    // Non-trap ("calm day") scenarios never touch the near-fragile links at all --
    // only bridge/ring incidents happen, so proximity siting is a genuinely sound
    // idea outside the published trap days. Trap-heavy scenarios sever EVERY
    // near-fragile link at once (the flood contour cutting all low-lying shortcut
    // roads simultaneously); their occasional "calm" sub-scenario is drawn from the
    // same nf-excluded pool.
    vector<char> isNf(Mblockable, 0);
    for (int d = 0; d < D; d++) isNf[nfIdx[d]] = 1;
    vector<int> nonNfPool;
    for (int e = 0; e < Mblockable; e++) if (!isNf[e]) nonNfPool.push_back(e);

    vector<vector<int>> scen(K);
    for (int s = 0; s < K; s++) {
        vector<int> blocked;
        if (trapHeavy && (s % 5 != 4)) {
            for (int d = 0; d < D; d++) blocked.push_back(nfIdx[d]);
            for (int r : ringIdx) if (rnd.next(100) < 30) blocked.push_back(r);
        } else {
            int prob = trapHeavy ? 15 : 10;
            for (int e : nonNfPool) if (rnd.next(100) < prob) blocked.push_back(e);
        }
        sort(blocked.begin(), blocked.end());
        blocked.erase(unique(blocked.begin(), blocked.end()), blocked.end());
        scen[s] = blocked;
    }

    // ---- print ----
    printf("%d %d %d %d %d\n", N, M, C, S, K);
    for (int i = 1; i <= N; i++) printf("%lld%c", pop[i], i == N ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d %lld\n", e.u, e.v, e.cap);
    for (int i = 0; i < C; i++) printf("%d %lld\n", candNode[i], candCap[i]);
    for (int s = 0; s < K; s++) {
        printf("%d", (int)scen[s].size());
        for (int e : scen[s]) printf(" %d", e);
        printf("\n");
    }

    return 0;
}
