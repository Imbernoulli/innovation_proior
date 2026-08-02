#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Tunnel Sweep: Clearing the Underlevel Network"  (generator)
// family: pursuit-on-tree-sweep
//
// Builds a rooted tree (node 1 = root, p_i < i for i=2..N) plus a per-node
// sentry cost c_i. Three structural modes chosen per testId:
//   RAND   : classic random recursive tree (uniform parent choice) -- sanity,
//            mostly low branching, no strong trap.
//   HUB    : several high-degree "hub" junctions, each hub's children given a
//            deliberately MIXED (x-requirement, sentry-cost) profile so that
//            sorting children by their own search-requirement x_child alone
//            (the natural greedy) badly misorders them relative to the
//            cost-aware optimum -- this is the required trap (>=3/10 cases).
//   NEEDLE : a hub with many small, cheap children plus ONE single outlier
//            child carrying a very high sentry cost relative to its x -- a
//            needle that greedy (sorts by x only, x of the needle is tiny so
//            it's scheduled early/whenever, ignoring its huge cost) buries in
//            the wrong place, forcing every sibling scheduled after it to
//            inherit a huge extra prefix cost.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    enum Mode { RAND, HUB, NEEDLE };
    Mode mode;
    int N;
    switch (testId){
        case 1:  mode = RAND;   N = 6;    break;
        case 2:  mode = HUB;    N = 12;   break;
        case 3:  mode = HUB;    N = 45;   break;
        case 4:  mode = NEEDLE; N = 70;   break;
        case 5:  mode = HUB;    N = 180;  break;
        case 6:  mode = RAND;   N = 320;  break;
        case 7:  mode = HUB;    N = 600;  break;
        case 8:  mode = NEEDLE; N = 1000; break;
        case 9:  mode = HUB;    N = 1900; break;
        default: mode = HUB;    N = 3000; break; // testId 10: fill the envelope
    }

    vector<int> par(N + 1, 0), cost(N + 1, 0);

    if (mode == RAND){
        for (int i = 2; i <= N; i++){
            par[i] = 1 + rnd.next(0, i - 2);
            cost[i] = rnd.next(1, 1000);
        }
    } else if (mode == HUB){
        // Grow the tree as a sequence of "hub bursts": pick an existing node,
        // attach a batch of children to it, and occasionally promote one of
        // the new children into a future hub (so hubs nest at different
        // depths, not just directly off the root).
        //
        // STAR-HUB TRAP: each hub keeps its OWN running counter across every
        // burst it ever receives (a hub can be revisited by several bursts,
        // e.g. leftover nodes near the size cap). Its k-th child (k=0,1,2,..
        // in id order, counted cumulatively for THAT hub, never reset) gets
        // sentry cost ~ 1000 - decay*k: strictly descending as id ascends.
        // Almost every child stays a leaf (x=1), so within one hub's child
        // set the x-values are ALL TIED and the true optimum is "highest
        // cost last". Greedy's tie-break (ascending id) does the opposite --
        // it schedules the priciest sentry FIRST, so every later sibling
        // inherits its huge cost. A per-hub (not per-burst) counter avoids
        // ever re-spiking back to cost~1000 partway through the same hub.
        vector<int> hubs = {1};
        vector<int> hubCounter(N + 1, 0);
        int next = 2;
        while (next <= N){
            int hub = hubs[rnd.next(0, (int)hubs.size() - 1)];
            int burst = min(N - next + 1, rnd.next(5, 14));
            for (int j = 0; j < burst && next <= N; j++, next++){
                par[next] = hub;
                int k = hubCounter[hub]++;
                int base = 1000 - 85 * k;
                int jitter = rnd.next(-10, 10);
                cost[next] = max(1, min(1000, base + jitter));
                if (rnd.next(0, 5) == 0) hubs.push_back(next);
            }
        }
    } else { // NEEDLE
        // Same star-hub construction, but every hub's FIRST child (in id
        // order) is a single extreme-cost needle (cost 1000) while the rest
        // of that hub's children are uniformly cheap -- an even sharper,
        // sparser version of the trap: one outlier per hub instead of a
        // smooth descending ramp.
        vector<int> hubs = {1};
        vector<int> hubCounter(N + 1, 0);
        int next = 2;
        while (next <= N){
            int hub = hubs[rnd.next(0, (int)hubs.size() - 1)];
            int burst = min(N - next + 1, rnd.next(6, 16));
            for (int j = 0; j < burst && next <= N; j++, next++){
                par[next] = hub;
                int k = hubCounter[hub]++;
                cost[next] = (k == 0) ? 950 : rnd.next(1, 60);
                if (rnd.next(0, 6) == 0) hubs.push_back(next);
            }
        }
    }

    printf("%d\n", N);
    for (int i = 2; i <= N; i++) printf("%d %d\n", par[i], cost[i]);
    return 0;
}
