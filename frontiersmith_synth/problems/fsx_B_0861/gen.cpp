#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Blind Delivery Through Road Closures"  (generator)
// family: static-detour-robust-route
//
// A road network on N intersections (1 = depot) has a spanning-tree BACKBONE
// (never blocked -- always drivable) plus EXTRA edges that individual closure
// scenarios may block. K closure scenarios are given, each a weighted set of
// blocked extra-edge ids. The participant prints a visiting order (permutation
// of stops 2..N); the checker replays every consecutive leg under every
// scenario using the FIXED rule "drive the shortest currently-open path" and
// scores a weighted sum of scenario travel totals.
//
// Three generator modes, chosen per testId to plant the required trap/needle
// structure (>=3 of 10 cases must punish naive nearest-neighbour heavily):
//   RANDOM  : generic random graph + random scenario blocks (sanity / no trap).
//   ARMHUB  : hub with two chain "arms"; matching-index TWIN shortcuts between
//             arms are individually cheap (< a chain hop) so raw-distance
//             nearest-neighbour ping-pongs across them constantly. Scenarios
//             block ranges of twins with real weight -> naive order pays a
//             detour on MANY legs; a smart order crosses arms once, at an
//             index chosen to minimise scenario-weighted exposure.
//   NEEDLE  : a mostly-random graph plus ONE very cheap long-range shortcut
//             edge (the "needle") that raw nearest-neighbour loves to use;
//             one heavy scenario blocks exactly that shortcut.
// -----------------------------------------------------------------------------

static vector<array<ll,3>> edges;     // {u, v, len}, 1-indexed nodes
static int N;

int addEdge(ll u, ll v, ll len){
    edges.push_back({u, v, len});
    return (int)edges.size();          // 1-based edge id
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    edges.clear();

    enum Mode { RANDOM, ARMHUB, NEEDLE };
    Mode mode;
    int m = 0;                          // arm length for ARMHUB
    int Nrand = 0, extraRand = 0;        // sizes for RANDOM / NEEDLE
    int K;

    switch (testId){
        case 1:  mode = RANDOM; Nrand = 6;   extraRand = 3;  K = 2;  break;
        case 2:  mode = RANDOM; Nrand = 12;  extraRand = 6;  K = 4;  break;
        case 3:  mode = ARMHUB; m = 5;                        K = 5;  break;
        case 4:  mode = NEEDLE; Nrand = 20;  extraRand = 9;   K = 5;  break;
        case 5:  mode = ARMHUB; m = 10;                       K = 6;  break;
        case 6:  mode = RANDOM; Nrand = 30;  extraRand = 15;  K = 8;  break;
        case 7:  mode = ARMHUB; m = 18;                       K = 8;  break;
        case 8:  mode = NEEDLE; Nrand = 55;  extraRand = 24;  K = 9;  break;
        case 9:  mode = ARMHUB; m = 40;                       K = 11; break;
        default: mode = ARMHUB; m = 65;                       K = 14; break; // testId 10: fill envelope
    }

    const ll CHAIN_W = 10;              // backbone chain hop weight (ARMHUB)
    vector<string> scenarioLines;        // buffered, printed AFTER header+edges

    auto emitScenario = [&](vector<int>& ids, int w){
        sort(ids.begin(), ids.end());
        ids.erase(unique(ids.begin(), ids.end()), ids.end());
        string l1 = to_string((int)ids.size()) + " " + to_string(w) + "\n";
        string l2;
        for (size_t i = 0; i < ids.size(); i++){
            l2 += to_string(ids[i]);
            l2 += (i + 1 < ids.size()) ? ' ' : '\n';
        }
        scenarioLines.push_back(l1 + l2);
    };

    if (mode == ARMHUB){
        N = 2*m + 1;                     // 1 = hub/depot; nodes are INTERLEAVED
        // node ids are interleaved (A1=2,B1=3,A2=4,B2=5,...) on purpose: ascending
        // index order must NOT coincide with "finish arm A, then arm B" -- otherwise
        // the checker's do-nothing identity-order baseline would already be near the
        // safe reference and the problem would fail to discriminate.
        auto nodeA = [&](int i){ return 2*i; };
        auto nodeB = [&](int i){ return 2*i + 1; };
        int A0 = 1;
        addEdge(A0, nodeA(1), CHAIN_W);
        addEdge(A0, nodeB(1), CHAIN_W);
        for (int i = 1; i < m; i++) addEdge(nodeA(i), nodeA(i+1), CHAIN_W);
        for (int i = 1; i < m; i++) addEdge(nodeB(i), nodeB(i+1), CHAIN_W);
        vector<int> twinId(m + 1, -1);
        for (int i = 1; i <= m; i++){
            ll w = rnd.next(2, (int)CHAIN_W - 1);      // 2 .. 9
            twinId[i] = addEdge(nodeA(i), nodeB(i), w);
        }
        for (int s = 1; s <= K; s++){
            vector<int> ids;
            if (s == 1){
                // "citywide" scenario: nearly every twin shortcut is out at once, so
                // no nearby unblocked twin can rescue a detour -- a tour that leans on
                // twins pays a real hub-length detour on almost every crossing leg.
                for (int i = 1; i <= m; i++) if (rnd.next(0,99) < 92) ids.push_back(twinId[i]);
                if (ids.empty()) ids.push_back(twinId[1]);
                emitScenario(ids, rnd.next(35, 50));
                continue;
            }
            int cls = s % 3;
            if (cls == 1){
                int lo = m/2 + 1, hi = m;
                for (int i = lo; i <= hi; i++) if (rnd.next(0,99) < 80) ids.push_back(twinId[i]);
                if (ids.empty()) ids.push_back(twinId[hi]);
                emitScenario(ids, rnd.next(25, 50));
            } else if (cls == 2){
                int lo = 1, hi = max(1, m/2);
                for (int i = lo; i <= hi; i++) if (rnd.next(0,99) < 70) ids.push_back(twinId[i]);
                if (ids.empty()) ids.push_back(twinId[lo]);
                emitScenario(ids, rnd.next(8, 22));
            } else {
                for (int i = 1; i <= m; i++) if (rnd.next(0,99) < 25) ids.push_back(twinId[i]);
                if (ids.empty()) ids.push_back(twinId[1 + rnd.next(0, m-1)]);
                emitScenario(ids, rnd.next(1, 10));
            }
        }
    } else {
        N = Nrand;
        for (int v = 2; v <= N; v++){
            int u = 1 + rnd.next(0, v - 2);
            addEdge(u, v, rnd.next(5, 60));
        }
        int backboneCount = N - 1;
        int needleId = -1;
        int extraFirst = backboneCount + 1;
        for (int e = 0; e < extraRand; e++){
            int u = 1 + rnd.next(0, N - 1), v = 1 + rnd.next(0, N - 1);
            while (v == u) v = 1 + rnd.next(0, N - 1);
            addEdge(u, v, rnd.next(15, 70));
        }
        if (mode == NEEDLE){
            int u = 2, v = N;
            needleId = addEdge(u, v, 2);
        }
        int extraLast = (int)edges.size();
        for (int s = 1; s <= K; s++){
            vector<int> ids;
            if (mode == NEEDLE && s == 1 && needleId != -1){
                ids.push_back(needleId);
                emitScenario(ids, rnd.next(30, 50));
                continue;
            }
            for (int e = extraFirst; e <= extraLast; e++)
                if (e != needleId && rnd.next(0, 99) < 30) ids.push_back(e);
            if (ids.empty()) ids.push_back(extraFirst + rnd.next(0, max(0, extraLast - extraFirst)));
            emitScenario(ids, rnd.next(1, 15));
        }
    }

    int M = (int)edges.size();
    int K_out = (int)scenarioLines.size();
    printf("%d %d %d\n", N, M, K_out);
    for (auto &e : edges) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    for (auto &s : scenarioLines) fputs(s.c_str(), stdout);
    return 0;
}
