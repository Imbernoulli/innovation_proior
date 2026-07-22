#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Seating the Feuding Clans" (generator)
// family: phase-aware-code-layout   (skin: seat quarrelsome guests at one long table)
//
// N guests are seated in ONE line (positions 1..N along a long banquet table).
// Every guest belongs to one of K "clans" (given explicitly, phase[i]). A guest
// list of M remembered exchanges ("who spoke to whom, how many times") is also
// given, u v c. Exchanges are overwhelmingly WITHIN a clan (each clan has its
// own dense chatter, generated as a hidden Hamiltonian "gossip chain" plus
// light noise) but a scattered set of moderate-weight cross-clan "bridge"
// remarks also exist -- the flat M-line list does NOT separate these by clan,
// they are mixed together in generation order.
//
// The trap: the natural first-attempt heuristic for "seat frequent talkers
// together" is a SHORT-SIGHTED nearest-neighbour walk -- start at some guest,
// repeatedly hop to whichever unseated guest you exchanged the most words
// with, and once you run out of exchange partners just seat the next unseated
// guest in id order and carry on (see solutions/greedy.cpp). Because each
// clan's dense chatter is a sparse Hamiltonian "gossip chain" (about degree 2
// per guest), this walk reconstructs a clan's path correctly ONLY as long as
// it keeps entering from a chain endpoint; the moment it runs dry mid-clan it
// falls back to id order, which -- because clan membership is well mixed
// across guest ids (see the Fisher-Yates shuffle below) -- drops it into a
// effectively random clan, often mid-chain, stranding the far half of that
// clan's path to be picked up later by more random id-order jumps. This
// fragments clans across the table (repeated changeover tax) and strands
// heavy in-clan pairs far apart (repeated distance-bucket penalties), even
// though the underlying gossip-chain structure it never noticed is a
// near-solved instance. The insight (solutions/strong.cpp): read the clan
// labels FIRST (un-mix the M-line list by clan), reconstruct one contiguous
// chain per clan directly from that clan's own exchange edges (no reliance on
// entering at the right endpoint), and only then decide the clan-block order.
// The scattered moderate-weight cross-clan "bridge" remarks exist so that the
// clan-block order is itself a nontrivial (if secondary) choice, not just
// noise.
// -----------------------------------------------------------------------------

struct Edge { int u, v, c; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int N, K;
    bool trap;                 // whether to inject the dense scattered-bridge trap
    switch (testId) {
        case 1:  N = 10;   K = 2; trap = false; break;
        case 2:  N = 16;   K = 2; trap = false; break;
        case 3:  N = 30;   K = 3; trap = false; break;
        case 4:  N = 60;   K = 3; trap = true;  break;
        case 5:  N = 120;  K = 4; trap = true;  break;
        case 6:  N = 260;  K = 4; trap = true;  break;
        case 7:  N = 520;  K = 5; trap = true;  break;
        case 8:  N = 1100; K = 5; trap = true;  break;
        case 9:  N = 2200; K = 6; trap = true;  break;
        default: N = 4200; K = 6; trap = true;  break; // testId 10: fill the envelope
    }

    // ---- assign each guest (1..N) to a clan, WELL MIXED so the identity order
    //      1,2,...,N is not already clan-contiguous (keeps the do-nothing
    //      baseline honest and non-trivial). Round-robin with a random per-clan
    //      size jitter, then a light shuffle of the assignment itself. ----
    vector<int> phase(N + 1, 0);
    {
        vector<int> order(N);
        for (int i = 0; i < N; i++) order[i] = i + 1;
        for (int i = N - 1; i > 0; i--) {           // Fisher-Yates via testlib rnd
            int j = rnd.next(0, i);
            swap(order[i], order[j]);
        }
        for (int i = 0; i < N; i++) phase[order[i]] = 1 + (i % K);
    }
    vector<vector<int>> members(K + 1);
    for (int g = 1; g <= N; g++) members[phase[g]].push_back(g);

    vector<Edge> edges;

    // ---- within-clan structure: a hidden "gossip chain" (Hamiltonian path over
    //      a random local permutation of the clan) with heavy weight, plus a
    //      sprinkle of lighter random within-clan noise edges. ----
    for (int k = 1; k <= K; k++) {
        vector<int> mem = members[k];
        int nk = (int)mem.size();
        if (nk < 2) continue;
        vector<int> perm = mem;
        for (int i = nk - 1; i > 0; i--) { int j = rnd.next(0, i); swap(perm[i], perm[j]); }
        for (int i = 0; i + 1 < nk; i++) {
            int w = rnd.next(80, 120);
            edges.push_back({perm[i], perm[i + 1], w});
        }
        int noiseCount = max(1, nk / 3);
        for (int t = 0; t < noiseCount; t++) {
            int a = mem[rnd.next(0, nk - 1)];
            int b = mem[rnd.next(0, nk - 1)];
            if (a == b) continue;
            int w = rnd.next(5, 20);
            edges.push_back({a, b, w});
        }
    }

    // ---- cross-clan bridge remarks: moderate weight, and on trap tests,
    //      numerous + scattered across many distinct guest pairs so the
    //      global weight-sorted chain heuristic keeps re-attaching across
    //      clan boundaries instead of finishing one clan before the next. ----
    int bridgeCount;
    if (!trap) bridgeCount = max(1, N / 10);
    else       bridgeCount = max(4, N / 3);
    for (int t = 0; t < bridgeCount; t++) {
        int k1 = 1 + rnd.next(0, K - 1);
        int k2 = 1 + rnd.next(0, K - 1);
        if (k1 == k2) k2 = 1 + (k2 % K);
        if (members[k1].empty() || members[k2].empty()) continue;
        int a = members[k1][rnd.next(0, (int)members[k1].size() - 1)];
        int b = members[k2][rnd.next(0, (int)members[k2].size() - 1)];
        int w = trap ? rnd.next(30, 60) : rnd.next(10, 25);
        edges.push_back({a, b, w});
    }

    // shuffle edge emission order (mixture: the list is NOT grouped by clan)
    for (int i = (int)edges.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(edges[i], edges[j]);
    }

    int M = (int)edges.size();
    printf("%d %d %d\n", N, K, M);
    for (int g = 1; g <= N; g++) printf("%d%c", phase[g], g < N ? ' ' : '\n');
    for (auto& e : edges) printf("%d %d %d\n", e.u, e.v, e.c);
    return 0;
}
