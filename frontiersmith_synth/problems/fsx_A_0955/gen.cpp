#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Placards Printed Once, Fires Vary"   family: evac-assignment-fire-scenarios
//
// Building = C independent CLUSTERS sharing a HUB, plus two exits:
//   exit0 ("main")  reachable only via   cluster -> hub -> exit0   (short, FAST)
//   exit1 ("bypass") reachable directly via  cluster -> exit1       (long, SLOW,
//                                             but a PRIVATE corridor per cluster)
// Every room belongs to exactly one cluster and connects ONLY to that cluster's
// junction node (one edge). So every room's route must go
//   room -> cluster -> hub -> exit0        (3 edges, short total transit weight)
//   room -> cluster -> exit1               (2 edges, but far longer transit weight)
//
// K fire scenarios each knock a cluster's HUB corridor or BYPASS corridor down to
// a tiny fraction of its normal throughput (occasionally the shared hub->exit0
// aggregator too). Every cluster's hub corridor gets hit hard in >=1 scenario and
// its bypass corridor gets hit hard in a DIFFERENT scenario, so once a cluster's
// population is large relative to either corridor's damaged capacity, no static
// all-or-nothing commitment is safe across the WHOLE ensemble: committing fully to
// hub eats the hub-damage scenario, committing fully to bypass eats the (separate)
// bypass-damage scenario, and SPLITTING the population bounds the worst of the two
// (space: fire falls on one specific corridor; time: whichever corridor is still
// open must swallow everyone's simultaneous throughput). "Nearest exit" (shortest
// weighted path per room) sends 100% of every cluster through the fast hub
// corridor regardless of scale -- exactly the corridor the ensemble is planted to
// burn. (On the smallest, tiny-population sanity cases the hub corridor's damaged
// capacity may still comfortably fit the whole cluster, in which case committing
// to hub is genuinely fine -- the ensemble's bite scales in with cluster size.)
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int C = 1 + (t >= 3 ? 1 : 0) + (t >= 6 ? 1 : 0) + (t >= 9 ? 1 : 0); // 1,1,2,2,2,3,3,3,4,4
    int roomsBase = 2 + t;                                              // 3..12

    vector<int> clusterOfRoom;              // room -> cluster (1-indexed)
    vector<vector<int>> roomsIn(C + 1);
    int N = 0;
    for (int c = 1; c <= C; c++){
        int rc = max(1, roomsBase + rnd.next(-1, 1));
        // "needle" clusters: cluster 1 gets an oversized population on selected tests
        if (c == 1 && (t == 5 || t == 8)) rc += roomsBase;
        for (int r = 0; r < rc; r++){ N++; clusterOfRoom.push_back(c); roomsIn[c].push_back(N); }
    }

    int hub = N + C + 1;
    int exit0 = N + C + 2;   // main, via hub
    int exit1 = N + C + 3;   // bypass, private per cluster
    int M = N + C + 3;
    int X = 2;

    // populations
    int popLo = 5, popHi = 10 + 2 * t;
    vector<ll> pop(N + 1, 0);
    vector<ll> clusterPop(C + 1, 0);
    for (int c = 1; c <= C; c++)
        for (int r : roomsIn[c]){
            ll base = rnd.next(popLo, popHi);
            if (c == 1 && (t == 5 || t == 8)) base *= 2;   // needle emphasis
            pop[r] = base;
            clusterPop[c] += base;
        }

    // edges: (u, v, cap, len)
    struct E{ int u, v; ll cap, len; };
    vector<E> edges;                                    // 0-indexed vector; edge id = index+1
    vector<int> hubEdgeOf(C + 1), bypEdgeOf(C + 1);

    ll capRoom = 1000000;
    for (int c = 1; c <= C; c++)
        for (int r : roomsIn[c])
            edges.push_back({r, c + N, capRoom, 1});     // room -> cluster junction

    for (int c = 1; c <= C; c++){
        ll divisor = rnd.next(3, 5);
        ll capHub = max(2LL, clusterPop[c] / divisor);
        ll lenHub = rnd.next(3, 6);
        edges.push_back({c + N, hub, capHub, lenHub});
        hubEdgeOf[c] = (int)edges.size();                // 1-indexed id
    }
    for (int c = 1; c <= C; c++){
        ll mult = rnd.next(10, 15);
        ll capByp = max(2LL, clusterPop[c] * mult / 10);
        ll lenByp = rnd.next(12, 20);
        edges.push_back({c + N, exit1, capByp, lenByp});
        bypEdgeOf[c] = (int)edges.size();
    }
    ll totalPop = 0; for (int c = 1; c <= C; c++) totalPop += clusterPop[c];
    ll capHubExit = max(2LL, totalPop);
    ll lenHubExit = rnd.next(2, 4);
    edges.push_back({hub, exit0, capHubExit, lenHubExit});
    int hubExitEdge = (int)edges.size();

    int Ecnt = (int)edges.size();

    // K fire scenarios: capacity-reduction damage (percent of original capacity kept)
    int K = 8;
    vector<vector<pair<int,int>>> scen(K + 1);
    for (int s = 1; s <= K; s++){
        if (C == 1){
            if (s % 2 == 1) scen[s].push_back({hubEdgeOf[1], rnd.next(3, 15)});
            else            scen[s].push_back({bypEdgeOf[1], rnd.next(3, 15)});
        } else {
            int hubC = ((s - 1) % C) + 1;
            scen[s].push_back({hubEdgeOf[hubC], rnd.next(3, 15)});
            int offset = 1 + rnd.next(0, C - 2);
            int bypC = ((hubC - 1 + offset) % C) + 1;
            scen[s].push_back({bypEdgeOf[bypC], rnd.next(3, 15)});
        }
        if (rnd.next(1, 100) <= 30)
            scen[s].push_back({hubExitEdge, rnd.next(50, 90)});
    }

    // ---- emit ----
    printf("%d %d %d %d %d\n", N, M, Ecnt, X, K);
    printf("%d %d\n", exit0, exit1);
    for (int i = 1; i <= N; i++) printf("%lld%c", pop[i], i == N ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d %lld %lld\n", e.u, e.v, e.cap, e.len);
    for (int s = 1; s <= K; s++){
        printf("%d\n", (int)scen[s].size());
        for (auto &pr : scen[s]) printf("%d %d\n", pr.first, pr.second);
    }
    return 0;
}
