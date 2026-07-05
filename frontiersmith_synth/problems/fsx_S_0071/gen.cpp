#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ----------------------------------------------------------------------------
// Recycling Depot Routes: Maximum-Weight Independent Set on a general conflict
// graph (family: independent-set-clique, variant 5, large scale).
//
// We emit a general conflict graph among candidate collection ROUTES:
//   * routes are partitioned into C service ZONES; every two routes in the same
//     zone conflict (a zone is a clique) -> at most one route per zone.
//   * a handful of PREMIUM (high-profit) routes per zone additionally conflict
//     with premium routes in OTHER zones (random cross-zone conflicts) -> you
//     cannot simply grab the most profitable route in every zone; picking a good
//     independent subset of premiums across zones is the hard combinatorial core.
//   * a sprinkle of random cross-zone conflicts among ordinary routes adds noise.
//
// testId is a difficulty/structure ladder: testId 1 is tiny (example scale),
// growing to a large, dense instance by testId 10.
// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int zones   = 2 + testId;                         // 3 .. 12 service zones
    int zsize   = 2 + (testId - 1) * 28;             // 2 .. 254 routes per zone
    int n       = zones * zsize;

    // premium routes per zone (high profit, cross-zone conflicts)
    int premPerZone = min(zsize, 2 + (testId % 3));   // 2 .. 4

    // profit ceilings vary per test so the "best single route" baseline shifts
    int ordHi   = 60 + (testId % 5) * 15;             // ordinary-profit ceiling 60..120
    int premLo  = 500 + (testId % 4) * 60;            // premium-profit floor
    int premHi  = premLo + 400;                       // premium-profit ceiling

    // probability that two premiums in different zones conflict
    double pcross = 0.35 + 0.03 * (testId % 5);        // 0.35 .. 0.47

    // ---- assign each route to a zone and a profit ----
    vector<int> zoneOf(n + 1);
    vector<long long> w(n + 1);
    vector<vector<int>> members(zones);
    {
        int id = 1;
        for (int z = 0; z < zones; z++) {
            for (int j = 0; j < zsize; j++) {
                zoneOf[id] = z;
                members[z].push_back(id);
                w[id] = rnd.next(1, ordHi);
                id++;
            }
        }
    }

    // pick premium routes per zone and boost their profit
    vector<vector<int>> premiums(zones);
    for (int z = 0; z < zones; z++) {
        vector<int> idx = members[z];
        shuffle(idx.begin(), idx.end());
        for (int c = 0; c < premPerZone && c < (int)idx.size(); c++) {
            int v = idx[c];
            premiums[z].push_back(v);
            w[v] = rnd.next(premLo, premHi);
        }
    }

    // ---- build edge set (dedup via a hash set of ordered pairs) ----
    // encode edge (a<b) as (long long)a * (n+1) + b
    unordered_set<long long> seen;
    seen.reserve((size_t)n * 4);
    vector<pair<int,int>> edges;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        long long key = (long long)a * (n + 1) + b;
        if (seen.insert(key).second) edges.push_back({a, b});
    };

    // zone cliques (intra-zone conflicts) -> at most one route per zone
    for (int z = 0; z < zones; z++) {
        auto& mem = members[z];
        int sz = (int)mem.size();
        for (int a = 0; a < sz; a++)
            for (int b = a + 1; b < sz; b++)
                addEdge(mem[a], mem[b]);
    }

    // premium cross-zone conflicts (the hard core)
    for (int z1 = 0; z1 < zones; z1++)
        for (int z2 = z1 + 1; z2 < zones; z2++)
            for (int a : premiums[z1])
                for (int b : premiums[z2])
                    if (rnd.next(0.0, 1.0) < pcross)
                        addEdge(a, b);

    // random ordinary cross-zone conflicts (noise)
    int noise = n;                                    // ~ n extra cross conflicts
    for (int e = 0; e < noise; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (zoneOf[a] == zoneOf[b]) continue;         // already covered by clique
        addEdge(a, b);
    }

    // ---- shuffle vertex labels so structure is not exposed by index order ----
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) { int j = rnd.next(1, i); swap(perm[i], perm[j]); }
    // perm[i] = new label of old vertex i
    vector<long long> wNew(n + 1);
    for (int i = 1; i <= n; i++) wNew[perm[i]] = w[i];
    for (auto& e : edges) { e.first = perm[e.first]; e.second = perm[e.second]; }

    shuffle(edges.begin(), edges.end());
    int m = (int)edges.size();

    // ---- output ----
    printf("%d %d\n", n, m);
    for (int i = 1; i <= n; i++) printf("%lld%c", wNew[i], i == n ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
