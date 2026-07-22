#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Valve Segmentation Under a Double-Burst Sweep"  (generator)
// family: valve-segment-isolation
//
// The network is a "caterpillar" tree: K backbone hub nodes 1..K joined in a
// PATH by K-1 "bridge" pipes (indices 1..K-1 -- these are the ONLY valve-
// eligible / candidate pipes). Every hub also owns several "branch" pipes
// (indices K..M) each ending in a fresh leaf junction and carrying a demand.
//
// PLANTED STRUCTURE (never labeled in the output -- a solver must recover it
// from the graph + candidate list):
//   - "flagship" hubs: exactly ONE of their branch pipes carries a huge demand
//     (thousands), the rest of the network carries modest demand (tens to a
//     few hundred) spread across many hubs ("mass" hubs).
//   TRAP: an obvious greedy sorts pipes by their OWN demand and spends the
//   valve budget fencing off flagship hubs (the pipes that visually look
//   scary). That protects a few large single numbers but leaves the huge
//   *mass* of many medium branch pipes as one giant, never-cut segment --
//   which is what the objective actually punishes (segment TOTAL demand, not
//   any one pipe's demand).
//   NEEDLE (testId >= 8): the double-burst list is laid out at EVENLY SPACED
//   hub boundaries along the backbone, so wherever a plain demand-balanced
//   partition would naturally place a cut, a double-burst pair straddles it --
//   ignoring the list and cutting there turns two "safe" segments into one
//   unsafe union.
//
// Output:
//   N M V C
//   c_1..c_C                (candidate/valve-eligible pipe indices, = 1..K-1)
//   u_k v_k d_k   for k=1..M (pipe k; k=1..K-1 are bridges, k=K..M are leaves)
//   L
//   i_t j_t       for t=1..L  (double-burst pairs, both indices are leaves)
// -----------------------------------------------------------------------------

struct Params {
    int K, V, L2, numFlag, minLeaf, maxLeaf;
    ll massLo, massHi, flagLo, flagHi;
    bool needle;
};

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const Params TABLE[11] = {
        /*0*/ {0,0,0,0,0,0,0,0,0,0,false},
        /*1*/ {4,   1, 1,  0, 2, 2,   20,  60,    0,    0, false},
        /*2*/ {8,   2, 3,  0, 2, 4,   20,  80,    0,    0, false},
        /*3*/ {15,  3, 5,  1, 3, 5,   20, 100, 3000, 5000, false},
        /*4*/ {25,  3, 8,  2, 4, 6,   20, 120, 3500, 6000, false},
        /*5*/ {50,  4, 10, 4, 4, 7,   30, 150, 4000, 8000, false},
        /*6*/ {80,  5, 15, 5, 5, 8,   30, 150, 4500, 9000, false},
        /*7*/ {110, 6, 20, 6, 5, 9,   30, 160, 5000, 9500, false},
        /*8*/ {140, 6, 30, 5, 5, 9,   30, 160, 5000, 9500, true },
        /*9*/ {170, 7, 40, 6, 6, 10,  30, 170, 5500, 9800, true },
        /*10*/{200, 8, 50, 8, 6, 11,  30, 180, 6000, 9990, true },
    };
    Params P = TABLE[testId];

    int K = P.K;
    // choose flagship hubs, spread evenly across [1,K]
    vector<char> isFlag(K + 1, 0);
    {
        set<int> chosen;
        for (int f = 0; f < P.numFlag; f++) {
            int pos = (int)((ll)(f + 1) * K / (P.numFlag + 1));
            pos = max(1, min(K, pos));
            while (chosen.count(pos)) pos = max(1, min(K, pos + 1));
            chosen.insert(pos);
        }
        for (int p : chosen) isFlag[p] = 1;
    }

    // bridge demands
    vector<ll> bridgeDem(K);   // bridgeDem[i] for bridge connecting hub i,i+1 (1-indexed i=1..K-1)
    for (int i = 1; i <= K - 1; i++) bridgeDem[i] = rnd.next(1, 5);

    // per-hub leaf pipe lists: (leafEdgeIdx placeholder later), demands
    vector<int> leafCount(K + 1);
    vector<vector<ll>> leafDem(K + 1);
    for (int i = 1; i <= K; i++) {
        int lc = (P.minLeaf == P.maxLeaf) ? P.minLeaf : rnd.next(P.minLeaf, P.maxLeaf);
        leafCount[i] = lc;
        leafDem[i].resize(lc);
        for (int j = 0; j < lc; j++) leafDem[i][j] = rnd.next(P.massLo, P.massHi);
        if (isFlag[i]) {
            int slot = rnd.next(0, lc - 1);
            leafDem[i][slot] = rnd.next(P.flagLo, P.flagHi);
        }
    }

    // assign node ids: hubs 1..K, then leaves K+1..N in hub-major order
    int nextNode = K + 1;
    vector<vector<int>> leafNode(K + 1);
    for (int i = 1; i <= K; i++) {
        leafNode[i].resize(leafCount[i]);
        for (int j = 0; j < leafCount[i]; j++) leafNode[i][j] = nextNode++;
    }
    int N = nextNode - 1;

    // assign edge indices: bridges 1..K-1, then leaves K..M in hub-major order
    vector<vector<int>> leafEdgeIdx(K + 1);
    int nextEdge = K; // first leaf edge index
    for (int i = 1; i <= K; i++) {
        leafEdgeIdx[i].resize(leafCount[i]);
        for (int j = 0; j < leafCount[i]; j++) leafEdgeIdx[i][j] = nextEdge++;
    }
    int M = nextEdge - 1;
    int C = K - 1;

    // double-burst pairs: pick pairs of (hub, leaf-slot) from distinct hubs
    vector<pair<int,int>> pairs;
    set<pair<int,int>> seen;
    if (P.needle) {
        for (int t = 1; t <= P.L2; t++) {
            int pos = (int)((ll)t * (K - 1) / (P.L2 + 1));
            pos = max(1, min(K - 1, pos));
            int hubA = pos, hubB = pos + 1;
            int ea = leafEdgeIdx[hubA][rnd.next(0, leafCount[hubA] - 1)];
            int eb = leafEdgeIdx[hubB][rnd.next(0, leafCount[hubB] - 1)];
            int a = min(ea, eb), b = max(ea, eb);
            if (a == b || seen.count({a, b})) continue;
            seen.insert({a, b});
            pairs.push_back({a, b});
        }
    } else {
        // NEARBY pairs only (span 1..3 hubs): physically these represent
        // correlated failures from shared trenching, so the two hubs are
        // close together -- this also keeps the tree-path a strong solver
        // would "pre-glue" between them SHORT (never a reason to fuse huge
        // swaths of the network just to dodge one listed pair).
        int tries = 0;
        while ((int)pairs.size() < P.L2 && tries < P.L2 * 20 + 50) {
            tries++;
            int hubA = rnd.next(1, K);
            int span = rnd.next(1, min(3, K - 1));
            int dir = rnd.next(0, 1) ? 1 : -1;
            int hubB = hubA + dir * span;
            if (hubB < 1) hubB = hubA + span;
            if (hubB > K) hubB = hubA - span;
            if (hubB < 1 || hubB > K || hubA == hubB) continue;
            int ea = leafEdgeIdx[hubA][rnd.next(0, leafCount[hubA] - 1)];
            int eb = leafEdgeIdx[hubB][rnd.next(0, leafCount[hubB] - 1)];
            int a = min(ea, eb), b = max(ea, eb);
            if (a == b || seen.count({a, b})) continue;
            seen.insert({a, b});
            pairs.push_back({a, b});
        }
    }
    int L = (int)pairs.size();

    // ---- emit ----
    printf("%d %d %d %d\n", N, M, P.V, C);
    for (int i = 1; i <= C; i++) printf("%d%c", i, i == C ? '\n' : ' ');
    if (C == 0) printf("\n");
    for (int i = 1; i <= K - 1; i++) printf("%d %d %lld\n", i, i + 1, bridgeDem[i]);
    for (int i = 1; i <= K; i++)
        for (int j = 0; j < leafCount[i]; j++)
            printf("%d %d %lld\n", i, leafNode[i][j], leafDem[i][j]);
    printf("%d\n", L);
    for (auto& pr : pairs) printf("%d %d\n", pr.first, pr.second);

    return 0;
}
