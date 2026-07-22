#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Ferry Ropes Across the Two Riverbank Towns"   family: skip-pointer-loom
//
// M ferry lanes, each a sorted list of mooring-post markers. Q convoys, each a
// set of 2-3 lanes that must be checked for joint layover posts by a FIXED
// pointer-walking dispatcher (see chk.cc). The solver strings up to B
// "shortcut ropes" (skip pointers), to be spent wherever they save the
// dispatcher the most walking (probes), across ALL convoys that share a lane.
//
// VALUE STRUCTURE: every lane starts with the SAME tiny "shared quay" prefix
// (values 1..S, identical across all lanes -- an easy first joint layover),
// then a large PRIVATE cluster of markers in a numeric slot that is disjoint
// from every other lane's private slot (slots are a shuffled, widely spaced
// partition of the value axis). Consequently, in any convoy, whichever named
// lane drew the LOWEST private slot must walk its *entire* private cluster,
// one marker at a time, before it can ever match again -- one long, single
// stall run per convoy, on the specific lane that happens to be "furthest
// behind" for THAT convoy. Which lane that is, and where the run starts,
// is a property of the ACTUAL convoy membership (found only by replaying the
// dispatcher), not of a lane's length or index proportion.
//
// PLANTED / TRAP structure:
//   - decoy lanes: a growing fraction of lanes are never touched by any
//     convoy -> length-proportional rope allocation wastes ropes there.
//   - hotspot lanes: a small subset of lanes appear in a large fraction of
//     convoys (cross-list-query-coupling) -> a SINGLE well-placed rope on a
//     hotspot lane's run start pays off in every convoy that names it.
//   - the run always starts right after the tiny shared prefix (position S),
//     essentially at the front of the lane -- a uniform "space ropes every
//     len/(k+1)" allocation puts its first breakpoint deep inside the
//     cluster instead, so it only ever collapses the TAIL of the run and
//     walks the (large) head one marker at a time.
//
// Output format (this generator prints ONE test):
//   M Q B
//   for c=1..M:  n_c
//                n_c strictly increasing integers (lane c's mooring posts)
//   for q=1..Q:  k  c_1 .. c_k        (k in {2,3}, distinct 1-indexed lanes)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;   // 0 .. 1

    int M = (int)llround(4 + f * 96.0);           // 4 .. 100
    int Q = (int)llround(3 + f * 597.0);          // 3 .. 600
    int S = 2 + (testId % 2);                     // 2 or 3 -- tiny shared prefix
    int hotLo = (int)llround(20 + f * 70.0);       // private-cluster size
    int hotHi = (int)llround(hotLo + 15 + f * 85.0);
    ll bigGap = (ll)(hotHi) * 3 + 300;              // guarantees disjoint slots

    double decoyFrac = (f < 0.35) ? 0.0 : (f - 0.35) / 0.65 * 0.55; // 0 .. 0.55
    int activeM = max(2, M - (int)llround(M * decoyFrac));
    if (activeM > M) activeM = M;
    int hotspotCount = max(2, (int)llround(activeM * 0.22));
    double hotspotQueryFrac = 0.20 + f * 0.30; // 0.20 .. 0.50
    int B = min(500, max(3, (int)llround(activeM * 0.58)));

    // ---- shuffled, disjoint value slots (one per lane) ----
    vector<int> slot(M);
    for (int i = 0; i < M; i++) slot[i] = i;
    for (int i = M - 1; i > 0; i--) swap(slot[i], slot[rnd.next(0, i)]);

    // ---- build lanes ----
    vector<vector<ll>> lane(M + 1);
    for (int c = 1; c <= M; c++) {
        int H = rnd.next(hotLo, hotHi);
        vector<ll> vals;
        for (int j = 1; j <= S; j++) vals.push_back(j);          // shared quay prefix
        ll cur = (ll)S + 1 + (ll)slot[c - 1] * bigGap + rnd.next(0, 20);
        for (int j = 0; j < H; j++) {
            vals.push_back(cur);
            cur += rnd.next(1, 3);
        }
        lane[c] = vals;
    }

    // ---- queries: pick from the active pool [1..activeM], biased to hotspots ----
    vector<int> hotspots;
    for (int i = 1; i <= hotspotCount; i++) hotspots.push_back(i);

    vector<int> qk(Q);
    vector<vector<int>> qch(Q);
    for (int q = 0; q < Q; q++) {
        int k = (rnd.next(1, 100) <= 72) ? 2 : 3;
        set<int> chosen;
        if (rnd.next(1, 100) <= (int)llround(hotspotQueryFrac * 100)) {
            chosen.insert(hotspots[rnd.next(0, (int)hotspots.size() - 1)]);
        }
        while ((int)chosen.size() < k) {
            int c = rnd.next(1, activeM);
            chosen.insert(c);
        }
        qk[q] = k;
        qch[q] = vector<int>(chosen.begin(), chosen.end());
    }

    // ---- print ----
    printf("%d %d %d\n", M, Q, B);
    for (int c = 1; c <= M; c++) {
        printf("%d\n", (int)lane[c].size());
        for (size_t j = 0; j < lane[c].size(); j++) printf("%lld%c", lane[c][j], j + 1 == lane[c].size() ? '\n' : ' ');
    }
    for (int q = 0; q < Q; q++) {
        printf("%d", qk[q]);
        for (int c : qch[q]) printf(" %d", c);
        printf("\n");
    }
    return 0;
}
