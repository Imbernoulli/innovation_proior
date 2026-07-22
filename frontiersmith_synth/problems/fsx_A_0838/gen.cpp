#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    if (testId < 1) testId = 1;
    if (testId > 10) testId = 10;

    // Zone count, pages-per-zone, and round count per test id: a size/difficulty ladder.
    // H (hub count) == pageCap always, so the hub group fills exactly one page when
    // clustered correctly. L = pz + 1 so a correct placement's zone-pages + hub-page fit
    // the cart exactly; an incorrect (graph-adjacency) placement thrashes on hub visits.
    int Zt[11]  = {0, 2, 3, 3, 4, 5, 6, 8, 10, 20, 40};
    int pzt[11] = {0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6};
    int Rt[11]  = {0, 2, 3, 4, 6, 10, 15, 25, 40, 80, 150};

    int Z = Zt[testId];
    int pz = pzt[testId];
    int R = Rt[testId];
    int pageCap = 8;
    int H = pageCap;
    int zoneSize = pageCap * pz;
    int n = Z * zoneSize + H;
    int L = pz + 1;

    // Logical layout: hubs are logical ids [0, H); zone z occupies
    // [H + z*zoneSize, H + (z+1)*zoneSize).
    auto zoneNode = [&](int z, int k) { return H + z * zoneSize + k; };

    // Random permutation from logical id -> printed animal id, so printed id order
    // carries NO information about zone/hub membership (defeats id-order baselines
    // from accidentally recovering structure).
    vector<int> id(n);
    for (int i = 0; i < n; i++) id[i] = i + 1;
    for (int i = n - 1; i > 0; i--) { int j = rnd.next(i + 1); swap(id[i], id[j]); }
    auto AID = [&](int logical) { return id[logical]; };

    // Paddock-adjacency graph: a path through each zone (recoverable zone structure via
    // BFS/DFS), one edge tying each hub to a random node of a random "home" zone (so a
    // graph-locality layout packs each hub with that zone, NOT with the other hubs),
    // plus a few cross-zone noise edges.
    vector<pair<int,int>> edges;
    for (int z = 0; z < Z; z++)
        for (int k = 0; k + 1 < zoneSize; k++)
            edges.push_back({AID(zoneNode(z, k)), AID(zoneNode(z, k + 1))});
    for (int h = 0; h < H; h++) {
        int homeZone = rnd.next(Z);
        int partner = zoneNode(homeZone, rnd.next(zoneSize));
        edges.push_back({AID(h), AID(partner)});
    }
    for (int t = 0; t < Z; t++) {
        int z1 = rnd.next(Z), z2 = rnd.next(Z);
        while (z2 == z1) z2 = rnd.next(Z);
        int a = AID(zoneNode(z1, rnd.next(zoneSize)));
        int b = AID(zoneNode(z2, rnd.next(zoneSize)));
        if (a != b) edges.push_back({a, b});
    }
    for (int i = (int)edges.size() - 1; i > 0; i--) { int j = rnd.next(i + 1); swap(edges[i], edges[j]); }

    // Fixed tour: R full cycles through all Z zones; within each zone visit, a shuffled
    // sweep of that zone's animals, followed immediately by a shuffled visit to every hub
    // ("the central rounds") -- the recurring inter-cluster shortcut the tour is built on.
    vector<int> trace;
    trace.reserve((size_t)R * Z * (zoneSize + H));
    for (int r = 0; r < R; r++) {
        for (int z = 0; z < Z; z++) {
            vector<int> zn(zoneSize);
            for (int k = 0; k < zoneSize; k++) zn[k] = zoneNode(z, k);
            for (int i = (int)zn.size() - 1; i > 0; i--) { int j = rnd.next(i + 1); swap(zn[i], zn[j]); }
            for (int v : zn) trace.push_back(AID(v));

            vector<int> hb(H);
            for (int k = 0; k < H; k++) hb[k] = k;
            for (int i = (int)hb.size() - 1; i > 0; i--) { int j = rnd.next(i + 1); swap(hb[i], hb[j]); }
            for (int v : hb) trace.push_back(AID(v));
        }
    }
    int T = (int)trace.size();

    printf("%d %d %d %d %d\n", n, (int)edges.size(), pageCap, L, T);
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
    for (int i = 0; i < T; i++) printf("%d%c", trace[i], (i + 1 == T ? '\n' : ' '));
    return 0;
}
