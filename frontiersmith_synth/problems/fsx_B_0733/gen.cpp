#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Express Tunnels to the Depot"  (generator)  family: hub-dilution-shortcuts
//
// Base network: a depot t=1, plus P "tied pairs" of long pendant arms (each pair
// has TWO arms of (nearly) equal length hanging off t) and F short pendant leaves
// hanging directly off t (mass filler / decoy anchors). This is a TREE.
//
// Candidate shortcuts (M of them, participant may add up to k):
//   DIRECT(tip,t)   -- wire one arm's tip straight to the depot.
//   CROSS(tipA,tipB)-- wire the two tips of ONE tied pair to each other.
//   DECOY           -- an edge between two already-close leaves (or a leaf and t):
//                       barely reduces anyone's resistance to t, pure mass tax.
//   NOISE           -- a few random extra pairs, for realism/robustness.
//
// PLANTED TRAP: within a tied pair (arms of equal length L), DIRECT fixes ONE tip
// down to ~1 but leaves its (still length-L) twin as the new global worst -- the
// worst-case objective barely moves, yet mass (and hence the objective, via the
// H=2mR decomposition) rose. CROSS parallels the two tied resistances L rms into
// one ~L/2, and helps BOTH endpoints for the SAME one edge of mass -- with a
// tight budget (k == number of pairs) only the resistance/mass RATIO view finds
// this; length-first ("wire the farthest node to the depot") greedily grabs the
// tempting DIRECT edges and gets outperformed or actively backfires.
//
// Output: n m0 t k M
//         m0 lines "u v"  (original tree edges)
//         M  lines "a b"  (candidate shortcut endpoints, 1-indexed, index=line#)
// -----------------------------------------------------------------------------

static int nextId;
static vector<pair<int,int>> edges;
static vector<int> leafIds;

int buildArm(int len){
    int prev = 1;
    int tip = -1;
    for (int i = 0; i < len; i++){
        int v = nextId++;
        edges.push_back({prev, v});
        prev = v;
        tip = v;
    }
    return tip;
}

int buildLeaf(){
    int v = nextId++;
    edges.push_back({1, v});
    leafIds.push_back(v);
    return v;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // -------- testId == 1: fully hand-verified tiny worked example --------
    if (testId == 1){
        // n=4  path 1-2-3-4 (t=1). k=1. candidates: (4,1) direct-good, (2,1) decoy-dup.
        printf("4 3 1 1 2\n");
        printf("1 2\n2 3\n3 4\n");
        printf("4 1\n2 1\n");
        return 0;
    }

    nextId = 2;
    edges.clear();
    leafIds.clear();

    vector<int> pairLens;   // one entry per tied pair (both arms of that pair share this length)
    int F = 0, k = 0, DEC = 0, NOISE = 0;
    int singleArm = 0;      // for testId==2 : one lone arm, no pair structure

    switch (testId){
        case 2: singleArm = 20;                   F = 12; k = 1; DEC = 5;  NOISE = 1; break;
        case 3: pairLens = {15};                 F = 30; k = 1; DEC = 4;  NOISE = 4; break;
        case 4: pairLens = {30};                 F = 55; k = 1; DEC = 6;  NOISE = 6; break;
        case 5: singleArm = 55;                   F = 30; k = 1; DEC = 10; NOISE = 1; break;
        case 6: pairLens = {60, 35};              F = 90; k = 1; DEC = 12; NOISE = 10; break;
        case 7: pairLens = {75};                  F = 145; k = 1; DEC = 25; NOISE = 15; break;
        case 8: pairLens = {95};                  F = 105; k = 1; DEC = 45; NOISE = 18; break;
        case 9: pairLens = {30, 24, 18, 12};      F = 130; k = 3; DEC = 60; NOISE = 20; break;
        case 10: pairLens = {35, 28, 21, 14};     F = 100; k = 3; DEC = 100; NOISE = 25; break;
        default: pairLens = {20}; F = 10; k = 1; DEC = 5; NOISE = 1; break;
    }

    vector<pair<int,int>> tipsOfPair; // (tipA, tipB) per pair
    if (singleArm > 0){
        int tip = buildArm(singleArm);
        tipsOfPair.push_back({tip, -1});
    } else {
        for (int len : pairLens){
            int a = buildArm(len);
            int b = buildArm(len);
            tipsOfPair.push_back({a, b});
        }
    }
    for (int i = 0; i < F; i++) buildLeaf();

    int n = nextId - 1;
    int t = 1;

    set<pair<int,int>> used;
    for (auto &e : edges) used.insert({min(e.first,e.second), max(e.first,e.second)});

    vector<pair<int,int>> cands;
    auto addCand = [&](int a, int b){
        if (a == b) return;
        auto key = make_pair(min(a,b), max(a,b));
        if (used.count(key)) return; // don't duplicate an existing pair (keeps M clean)
        used.insert(key);
        cands.push_back({a, b});
    };

    // DIRECT + CROSS candidates
    for (auto &tp : tipsOfPair){
        addCand(tp.first, t);
        if (tp.second != -1){
            addCand(tp.second, t);
            addCand(tp.first, tp.second);
        }
    }

    // DECOY candidates: reinforce the already-close leaves (or leaf<->t duplicate)
    for (int d = 0; d < DEC; d++){
        if ((int)leafIds.size() >= 2 && rnd.next(0, 3) != 0){
            int i = rnd.next(0, (int)leafIds.size() - 1);
            int j;
            do { j = rnd.next(0, (int)leafIds.size() - 1); } while (j == i);
            addCand(leafIds[i], leafIds[j]);
        } else if (!leafIds.empty()){
            int i = rnd.next(0, (int)leafIds.size() - 1);
            addCand(leafIds[i], t);
        }
    }

    // NOISE candidates: a few uniformly random pairs, for realism
    for (int i = 0; i < NOISE; i++){
        int a = rnd.next(2, n);
        int b = rnd.next(1, n);
        addCand(a, b);
    }

    // Make sure we always have at least k candidates (pad with random pairs if a
    // tiny/edge-case parameter combo came up short).
    int guard = 0;
    while ((int)cands.size() < k && guard < 100000){
        int a = rnd.next(2, n);
        int b = rnd.next(1, n);
        addCand(a, b);
        guard++;
    }

    for (int i = (int)cands.size() - 1; i > 0; i--) swap(cands[i], cands[rnd.next(0, i)]);

    int m0 = (int)edges.size();
    int M = (int)cands.size();
    printf("%d %d %d %d %d\n", n, m0, t, k, M);
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
    for (auto &c : cands) printf("%d %d\n", c.first, c.second);
    return 0;
}
