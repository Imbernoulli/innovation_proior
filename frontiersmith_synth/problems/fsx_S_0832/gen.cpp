#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Interdependent power/comm coupling-design instance.
//
// Builds TWO graphs on the same site set 1..N:
//   P (power grid)      -- a cycle backbone + random chords + a top tier of
//                           high-degree "hub" sites plus a moderately-reinforced
//                           "backup" tier. The cycle backbone gives P redundancy:
//                           losing a handful of nodes (even hubs) does NOT
//                           fragment P's own giant component much.
//   C (comm/SCADA net)   -- a small number of "relay hub" nodes, each the star
//                           centre of its own cluster, hubs linked by a thin
//                           backbone cycle, plus a light sprinkling of redundancy
//                           edges. C is FRAGILE: losing a relay hub disconnects
//                           most of its cluster.
// C's relay hubs are PARTLY drawn from the same site positions as P's power
// hub/backup tier (~55% overlap) -- physical importance correlates a bit across
// co-located layers, so even the "pair site i with site i" baseline carries real
// hub-collision risk, without being fully equivalent to a deliberate degree sort.
// Both graphs are built on internal indices 0..N-1, then relabelled by a SINGLE
// shared random permutation before printing (so "site i" means the same physical
// location in both layers), hiding hub identity behind degree, never the raw id.
//
// Attack scenarios come in three families:
//   TARGETED    -- strikes a fraction of the important power sites (top hub tier
//                  + backup tier), most damaging to degree-correlated coupling.
//   STORM       -- broad random damage that spares the top hub tier entirely.
//   UNTARGETED  -- uniformly random damage, hubs included with the same odds.
// testId is a difficulty/structure ladder: N grows 24 -> 320 (filling the stated
// envelope by testId 10), and the scenario mix shifts along the ladder --
// testId 1..7 are STORM/UNTARGETED-only (degree-correlated coupling is genuinely
// good there), testId 8..10 are TARGETED-heavy trap tests (the same coupling
// becomes an outright liability). Nothing in the statement marks which regime a
// given test is in -- a solver has to read the attack scenarios themselves.

int N;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10]    = {24, 45, 70, 100, 140, 180, 220, 260, 300, 320};
    N = Ns[idx - 1];

    int Hp = max(2, N / 12);           // # power hubs
    int Hc = Hp;                       // # comm clusters/hubs
    int Dhub = max(3, N / 3);          // extra links per power hub

    // ---------- build P on internal indices 0..N-1 ----------
    set<pair<int,int>> edgesP;
    auto addEdgeP = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        edgesP.insert({a, b});
    };
    for (int i = 0; i < N; i++) addEdgeP(i, (i + 1) % N);          // cycle backbone
    int extraChords = max(1, N / 3);
    for (int t = 0; t < extraChords; t++) {
        int a = rnd.next(0, N - 1), b = rnd.next(0, N - 1);
        addEdgeP(a, b);
    }
    vector<int> hubOrderP(N);
    for (int i = 0; i < N; i++) hubOrderP[i] = i;
    shuffle(hubOrderP.begin(), hubOrderP.end());
    vector<int> hubsP(hubOrderP.begin(), hubOrderP.begin() + Hp);
    for (int h : hubsP) {
        int links = Dhub;
        for (int t = 0; t < links; t++) {
            int b = rnd.next(0, N - 1);
            addEdgeP(h, b);
        }
    }
    // A second tier of moderately-reinforced substations: NOT directly targeted
    // by any attack scenario, but noticeably more redundant than a plain
    // periphery node -- these are the sites a coupling can safely lean on to
    // shelter a fragile comm relay hub from broad storm damage without walking
    // into the deliberate attack on the top-tier hubs.
    int Hp2 = max(2, Hc);
    vector<int> backupP(hubOrderP.begin() + Hp, hubOrderP.begin() + min(N, Hp + Hp2));
    for (int h : backupP) {
        int links = max(1, Dhub / 2);
        for (int t = 0; t < links; t++) {
            int b = rnd.next(0, N - 1);
            addEdgeP(h, b);
        }
    }

    // ---------- build C on internal indices 0..N-1 ----------
    // IMPORTANT: C's relay hubs PARTLY sit at the same internal positions as
    // P's power hubs (a major substation site is somewhat likelier to also
    // host a major comm relay -- physical importance correlates a bit across
    // co-located infrastructure layers, but far from perfectly). Combined with
    // a single SHARED relabelling permutation for both P and C below, this
    // means the "do nothing, just pair site i with site i" baseline carries a
    // real (not merely accidental) share of hub-hub risk, without making it
    // fully equivalent to the deliberate sort-by-degree coupling.
    set<pair<int,int>> edgesC;
    auto addEdgeC = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        edgesC.insert({a, b});
    };
    vector<int> hubsC;
    {
        int overlap = max(1, Hc * 55 / 100);
        vector<int> hp = hubsP;
        hp.insert(hp.end(), backupP.begin(), backupP.end());
        shuffle(hp.begin(), hp.end());
        for (int i = 0; i < overlap && i < (int)hp.size(); i++) hubsC.push_back(hp[i]);
        set<int> used(hubsC.begin(), hubsC.end());
        vector<int> others;
        for (int i = 0; i < N; i++) if (!used.count(i)) others.push_back(i);
        shuffle(others.begin(), others.end());
        for (int i = 0; (int)hubsC.size() < Hc && i < (int)others.size(); i++) hubsC.push_back(others[i]);
    }
    // assign every non-hub node to a random cluster hub (star)
    vector<int> clusterOf(N, -1);
    for (int h : hubsC) clusterOf[h] = h;
    for (int i = 0; i < N; i++) {
        if (clusterOf[i] != -1) continue;
        int h = hubsC[rnd.next(0, Hc - 1)];
        clusterOf[i] = h;
        addEdgeC(i, h);
    }
    // link hubs together in a thin backbone cycle so C starts fully connected
    for (int t = 0; t < Hc; t++) addEdgeC(hubsC[t], hubsC[(t + 1) % Hc]);
    // light redundancy: a few extra random-in-cluster edges (not enough to save a
    // cluster once its hub dies)
    int extraC = max(1, N / 5);
    for (int t = 0; t < extraC; t++) {
        int a = rnd.next(0, N - 1), b = rnd.next(0, N - 1);
        addEdgeC(a, b);
    }

    // ---------- attack scenarios on P's internal indices ----------
    // Three scenario families:
    //   TARGETED  -- deliberately strikes (a subset of) the primary power hubs.
    //                Punishes routing a hub's dependency onto a comm hub.
    //   STORM     -- broad damage to ordinary substations that spares BOTH power
    //                hub tiers (built to a higher standard). Rewards routing a
    //                comm hub's dependency onto a resilient power site.
    //   UNTARGETED -- uniformly random damage, hubs included with the same odds
    //                as anyone else; favours neither rule on its own.
    // The mix shifts along the difficulty ladder: early (small-N) tests lean
    // STORM-heavy (a plain "keep well-connected things together" instinct
    // mostly pays off), late (large-N) tests lean TARGETED-heavy (that same
    // instinct becomes an outright liability) -- exactly the regime change the
    // solver has to notice from the graphs themselves, since nothing in the
    // statement says which regime a given test is in.
    vector<vector<int>> attacks;
    vector<int> importantP = hubsP;
    importantP.insert(importantP.end(), backupP.begin(), backupP.end());
    set<int> hubSetP(hubsP.begin(), hubsP.end());
    vector<int> nonHub;
    for (int i = 0; i < N; i++) if (!hubSetP.count(i)) nonHub.push_back(i);

    auto targeted = [&](int fracPct) {
        // A deliberate strike on the important power sites -- primary hubs
        // AND the reinforced backup tier alike, not just the very top rank.
        vector<int> hp = importantP;
        shuffle(hp.begin(), hp.end());
        int take = max(1, (int)hp.size() * fracPct / 100);
        attacks.push_back(vector<int>(hp.begin(), hp.begin() + min(take, (int)hp.size())));
    };
    auto storm = [&](int fracPct) {
        vector<int> pool = nonHub;
        shuffle(pool.begin(), pool.end());
        int cnt = max(1, (int)pool.size() * fracPct / 100);
        attacks.push_back(vector<int>(pool.begin(), pool.begin() + min(cnt, (int)pool.size())));
    };
    auto untargeted = [&](int fracPct) {
        vector<int> pool(N); for (int i = 0; i < N; i++) pool[i] = i;
        shuffle(pool.begin(), pool.end());
        int cnt = max(1, N * fracPct / 100);
        attacks.push_back(vector<int>(pool.begin(), pool.begin() + min(cnt, (int)pool.size())));
    };

    // A large, fixed bank of scenarios sampled every test (averages away
    // single-draw noise). testId 1..7: no deliberate hub targeting at all --
    // only broad storm/untargeted damage, which a well-connected pairing
    // rides out. testId 8..10 (the last 3, trap-dominant): mostly deliberate
    // hub-targeted strikes, which punish exactly that same "connect the
    // important things together" instinct. Nothing in the statement marks
    // which regime a given test is in -- the solver must infer it from the
    // graphs (and from how a candidate coupling behaves) itself.
    if (idx <= 7) {
        storm(25); storm(35); storm(45); storm(55); storm(65); storm(75);
        storm(83); storm(90); storm(95); storm(98);
        storm(50); storm(62); storm(72); untargeted(20);
    } else {
        targeted(45); targeted(60); targeted(72); targeted(84); targeted(94); targeted(100);
    }

    // ---------- relabel with a SINGLE shared permutation (see note above): P
    // and C are printed under the same site numbering, so "site i" means the
    // same physical location in both layers, and hub identity can only be
    // read off from degree, never from the raw id.
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());

    printf("%d %d %d\n", N, (int)edgesP.size(), (int)edgesC.size());
    for (auto& e : edgesP) printf("%d %d\n", perm[e.first] + 1, perm[e.second] + 1);
    for (auto& e : edgesC) printf("%d %d\n", perm[e.first] + 1, perm[e.second] + 1);
    printf("%d\n", (int)attacks.size());
    for (auto& atk : attacks) {
        printf("%d", (int)atk.size());
        for (int v : atk) printf(" %d", perm[v] + 1);
        printf("\n");
    }
    return 0;
}
