#include "testlib.h"
#include <vector>
#include <set>
#include <utility>
#include <algorithm>
using namespace std;

// One test file = a batch of K independent "arenas". Each arena is an avoidance
// claiming game on n vertices with poison edges. Vertices 1..n form n/2 base
// pairs (2i-1,2i); each base pair edge is ALWAYS a poison edge (listed first in
// the edge list for that arena). On top of that:
//  - "trap" arenas add a HUB: vertex 1 gets an extra poison edge to a random
//    ~40% of the other vertices, and the arena is forced to start with You
//    moving first (so You predictably claim the hub on turn 1). Blindly
//    mirroring the opponent afterward (claim the FIRST-edge partner of
//    whatever the opponent just took) has a high per-turn chance of claiming
//    a hub neighbor -> instantly completing a poison edge with the hub You
//    already own. A policy that reasons about what it already owns avoids
//    this entirely.
//  - other arenas add a light, unstructured sprinkling of cross edges
//    between different base pairs for general (non-adversarial) difficulty.

static void emitArena(int n, int first, const vector<pair<int,int>>& edges) {
    printf("%d %d %d\n", n, (int)edges.size(), first);
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int K = 6 + testId;                    // 7 .. 16 arenas per file
    int nMaxBase = 6 + 3 * testId;          // grows to 36
    bool trapFile = (testId >= 5);          // files 5..10 are trap-heavy (>=3 of 10 required)

    printf("%d\n", K);

    for (int a = 0; a < K; a++) {
        int pairs, first;
        vector<pair<int,int>> edges;
        set<pair<int,int>> edgeset;

        int numAnchors = max(2, K / 3);
        if (a < numAnchors) {
            // guaranteed tiny, cross-edge-free anchor arenas (baseline stability insurance)
            pairs = 1 + (a % 2);
            first = rnd.next(0, 1);
            for (int i = 1; i <= pairs; i++) edges.push_back({2 * i - 1, 2 * i});
        } else {
            int maxPairs = max(2, nMaxBase / 2);
            pairs = rnd.next(2, maxPairs);
            // base-pair (matching) edges -- always listed first
            for (int i = 1; i <= pairs; i++) {
                int u = 2 * i - 1, v = 2 * i;
                edges.push_back({u, v});
                edgeset.insert({u, v});
            }
            bool trapArena = trapFile && (rnd.next(0, 99) < 75);
            int n = pairs * 2;
            if (trapArena) {
                first = 0; // You always move first, so You predictably claim the hub (vertex 1)
                for (int w = 3; w <= n; w++) {
                    if (rnd.next(0, 99) < 42) {
                        int lo = min(1, w), hi = max(1, w);
                        if (edgeset.count({lo, hi})) continue;
                        edgeset.insert({lo, hi});
                        edges.push_back({1, w});
                    }
                }
            } else {
                first = rnd.next(0, 1);
                int crossTarget = pairs / 6;
                for (int c = 0; c < crossTarget; c++) {
                    for (int attempt = 0; attempt < 25; attempt++) {
                        int i = rnd.next(1, pairs);
                        int j = rnd.next(1, pairs);
                        if (i == j) continue;
                        int u = (rnd.next(0, 1) == 0) ? (2 * i - 1) : (2 * i);
                        int v = (rnd.next(0, 1) == 0) ? (2 * j - 1) : (2 * j);
                        int lo = min(u, v), hi = max(u, v);
                        if (edgeset.count({lo, hi})) continue;
                        edgeset.insert({lo, hi});
                        edges.push_back({u, v});
                        break;
                    }
                }
            }
        }
        int n = pairs * 2;
        emitArena(n, first, edges);
    }
    return 0;
}
