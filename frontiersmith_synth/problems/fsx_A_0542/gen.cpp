#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Grid Spine Under Targeted Outages"  (generator)
// family: failure-mask-backbone-sparsifier    objective: MINIMIZE
//
// Mechanisms composed into one objective:
//   (1) budgeted-graph-sparsification   -- keep at most Bbudget links.
//   (2) planted-cut-failure-scenarios   -- each scenario destroys ONE interior spine
//                                           link (a low-redundancy cut of the spine).
//   (3) stretch-plus-connectivity-penalty -- per scenario & demand you pay P if severed,
//                                           else the weighted shortest-path distance.
//
// STRUCTURE (the checker never sees these labels; it only replays shortest paths):
//   The spine is a single path threaded through C equal "blocks" (clusters). Block j
//   spans hubs [j*S, (j+1)*S] via S spine links; its demand pair is exactly (j*S,(j+1)*S).
//   Destroying ANY interior spine link of block j severs that block's pair.
//   Reinforcements are of two kinds:
//     * NARROW  -- a spare link parallel to one interior spine link (weight = spine).
//                  It heals exactly that ONE cut.
//     * WIDE    -- a direct link joining a block's two ends (weight Dd > block length).
//                  It heals EVERY interior cut of that block, but routes at a detour.
//   Each block is attacked by cpc=3 scenarios, each hitting a distinct interior link.
//
// THE TRAP: the obvious "add a backup for every failed link" heuristic buys narrows,
//   one per scenario; with Bbudget = M + 2C it patches only 2C of the 3C scenarios and
//   leaves C severed. The insight: the whole scenario family only attacks links INSIDE
//   the blocks, so one wide link per block is a size-C hitting set that heals all 3C
//   scenarios -- well within budget.
//
// P is calibrated per-instance so the spine-only baseline B is ~8x the hitting-set
// objective (strong ratio ~0.8, leaving headroom above the reference).
//
// Output:  n m M K D Bbudget P ; m links (u v w) ; D demands (s e) ; K scenarios (c d...).
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int C  = 3  + (int)llround(f * 13.0);   // 3..16  blocks
    int S  = 20 + (int)llround(f * 80.0);   // 20..100 spine links per block
    ll wseg = 1;                            // spine / narrow link weight
    ll Db  = S;                             // block base distance (unused directly)
    ll Dd  = S + S / 2;                     // wide detour weight (> block length)
    int M  = C * S;                         // spine links, indices 0..M-1
    int n  = M + 1;
    int cpc = 3;                            // scenarios (cuts) per block
    int K  = cpc * C;
    int D  = C;
    ll P   = 7LL * (C - 1) * S + 8LL * Dd;  // calibrated disconnection penalty
    int Bbudget = M + 2 * C;                // affords 2C reinforcements
    (void)Db;

    struct Edge { ll u, v, w; };
    vector<Edge> E;
    // spine links 0..M-1  (hub i -- hub i+1)
    for (int i = 0; i < M; i++) E.push_back({(ll)i, (ll)(i + 1), wseg});
    // WIDE links: M .. M+C-1  (block ends joined directly)
    for (int j = 0; j < C; j++) E.push_back({(ll)j * S, (ll)(j + 1) * S, Dd});
    // NARROW links + scenarios, grouped block-major
    vector<vector<int>> scen;               // each scenario: list of destroyed spine indices
    for (int j = 0; j < C; j++){
        set<int> offs;
        while ((int)offs.size() < cpc) offs.insert(rnd.next(0, S - 1));
        for (int off : offs){
            int cutIdx = j * S + off;       // spine link index (interior of block j)
            ll u = E[cutIdx].u, v = E[cutIdx].v;
            E.push_back({u, v, wseg});      // narrow spare parallel to that spine link
            scen.push_back({cutIdx});
        }
    }
    int m = (int)E.size();                  // M + C + 3C = M + 4C

    printf("%d %d %d %d %d %d %lld\n", n, m, M, K, D, Bbudget, P);
    for (auto& e : E) printf("%lld %lld %lld\n", e.u, e.v, e.w);
    for (int j = 0; j < C; j++) printf("%d %d\n", j * S, (j + 1) * S);
    for (auto& s : scen){
        printf("%d", (int)s.size());
        for (int d : s) printf(" %d", d);
        printf("\n");
    }
    return 0;
}
