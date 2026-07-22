#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator -- family: byproduct-symbiosis-loops
// Theme: eco-industrial park waste-exchange network.
//
// A directed exchange network. Feasibility (checked by chk.cc) is a CIRCULATION:
// material balance (inflow==outflow) at every plant, so all routed flow lives on
// closed loops. Objective F = sum rev_e * x_e (maximize).
//
// PLANTED STRUCTURE (disjoint node blocks, so each is chordless & edge-disjoint):
//   * BAIT 2-cycles   {a<->b}: net +4/unit, cap 20  -> value 80 each. These, and
//     only these, form the checker's bilateral baseline B. (byproduct-matching)
//   * 3-cycles        a->b->c->a: net +12/unit, cap 20 -> value 240 each.
//   * "captured" loops length 6..9 (per-edge rev +5, cap 20).
//   * "over-long" loops length 11..14 (per-edge rev +5, cap 20)  -- these carry the
//     MOST value per loop and are exactly what a pairwise/short-cycle matcher misses.
//     (planted-cycle-structure)
//   * NOISE edges with strongly NEGATIVE rev (-20..-60): a transport tax so heavy
//     that no cycle through a noise edge is worth routing. (waste-penalty)
//
// TRAP: the obvious "best-pair bilateral matching" captures only the 2-cycles (ratio
// ~0.1); a 3-cycle greedy reaches ~0.25; the value that dominates 10*B lives in the
// long chordless loops, reachable only by cycle-centric search (negative-cost cycle
// canceling on the residual graph).
//
// Node labels are randomly permuted and the edge list is shuffled, so the planted
// structure is not readable from the ordering.
// -----------------------------------------------------------------------------

struct E { int u, v; ll cap, rev; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int m      = 10 + (int)llround(f * 110.0);   // master scale 10..120
    int n2     = m;                              // bait 2-cycles
    int n3     = (int)llround(m * 0.5);          // 3-cycles
    int nLong  = (int)llround(m * 0.75);         // total long loops
    int nLongC = (int)llround(nLong * 0.6);      // captured (len 6..9)
    int nLongO = nLong - nLongC;                 // over-long (len 11..14)
    const ll CAP = 20;

    vector<E> edges;
    int nid = 0;  // 0-based node allocator; relabeled at output

    // captured loops, length 6..9
    for (int i = 0; i < nLongC; i++) {
        int L = 6 + rnd.next(0, 3);
        int base = nid;
        for (int k = 0; k < L; k++) {
            int a = base + k, b = base + (k + 1) % L;
            edges.push_back({a, b, CAP, 5});
        }
        nid += L;
    }
    // over-long loops, length 11..14 (dominant value, missed by shallow search)
    for (int i = 0; i < nLongO; i++) {
        int L = 11 + rnd.next(0, 3);
        int base = nid;
        for (int k = 0; k < L; k++) {
            int a = base + k, b = base + (k + 1) % L;
            edges.push_back({a, b, CAP, 5});
        }
        nid += L;
    }
    // 3-cycles, net +12
    for (int i = 0; i < n3; i++) {
        int a = nid, b = nid + 1, c = nid + 2;
        edges.push_back({a, b, CAP, 4});
        edges.push_back({b, c, CAP, 4});
        edges.push_back({c, a, CAP, 4});
        nid += 3;
    }
    // bait 2-cycles, net +4  (the whole baseline B)
    for (int i = 0; i < n2; i++) {
        int a = nid, b = nid + 1;
        edges.push_back({a, b, CAP, 2});
        edges.push_back({b, a, CAP, 2});
        nid += 2;
    }

    int structNodes = nid;
    int extra = (int)llround(f * 1500.0);   // idle plants to fill the envelope
    int N = structNodes + extra;
    if (N < 2) N = 2;

    // noise: strongly-taxed pipes (never worth routing on their own)
    int nNoise = (int)llround(f * 15000.0);
    for (int i = 0; i < nNoise; i++) {
        int u = rnd.next(0, N - 1), v = rnd.next(0, N - 1);
        if (u == v) v = (v + 1) % N;
        ll rv = -(20 + rnd.next(0, 40));   // -20..-60
        ll cp = 1 + rnd.next(0, 29);       // 1..30
        edges.push_back({u, v, cp, rv});
    }

    // random relabel of plant ids
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i;
    for (int i = N - 1; i > 0; i--) { int j = rnd.next(0, i); swap(perm[i], perm[j]); }
    // shuffle edge listing order
    for (int i = (int)edges.size() - 1; i > 0; i--) { int j = rnd.next(0, i); swap(edges[i], edges[j]); }

    int M = (int)edges.size();
    printf("%d %d\n", N, M);
    for (auto& e : edges)
        printf("%d %d %lld %lld\n", perm[e.u] + 1, perm[e.v] + 1, e.cap, e.rev);
    return 0;
}
