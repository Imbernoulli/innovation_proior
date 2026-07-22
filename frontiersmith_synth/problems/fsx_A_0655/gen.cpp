#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Star Lattice Interfaces"  (generator)   family: zellij-star-lattice
//
// Global constant L = 24 (the plane's angle unit modulus; every fixed edge angle
// a_e lies in [0,24)). A cell built with star-order k (k | 24) and rotation r
// presents strand angles = { (r + t*(24/k)) mod 24 : t=0..k-1 }; equivalently a
// cell is buildable with order k iff ALL of its incident edge angles are
// congruent to the SAME value modulo step=24/k (and r is then that residue).
//
// PLANTED STRUCTURE: the graph is a chain of Z "zones". Every star cell inside
// zone z has ALL of its incident edges tagged with the SAME angle A_z, so a star
// cell is always buildable under any of the 4 palette orders {4,6,8,12} -- the
// real decision is WHICH order to pick, to maximize order-diversity D and the
// canonical-phase bonus (mixed-rotational-orders + edge-angle-matching).
//
// Consecutive zones z, z+1 are bridged by exactly one ADAPTER cell whose own
// incident edges mix angle A_z (edges into zone z) and A_{z+1} (edges into
// zone z+1) -- two GENUINELY different residues that no fixed small order can
// reconcile in general. Only an adapter (allowed any k in {3,4,6,8,12,24}, a
// "common refinement" vocabulary) can be built there, and its CHEAPEST valid
// order is the number-theoretic minimum k such that step=24/k divides
// gcd(A_{z+1}-A_z, 24) (adapter-tile-design). The zone-angle deltas cycle
// through 4 categories (multiples of 8, of 6, of 4, and values coprime to 24)
// so required adapter orders range from cheap (k=3) to the expensive universal
// k=24 -- every test case includes this trap: greedy blindly using k=24
// everywhere overpays badly on the cheap junctions.
//
// Output:
//   N M
//   N ints: type_i (0=star,1=adapter)
//   p
//   p pairs: k C   (star palette order -> canonical phase in [0,24/k))
//   W1 W2 W3
//   N ints: cost_i (adapter unit cost; 0 for star cells)
//   M lines: u v a len
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    const int L = 24;
    const int paletteK[4] = {4, 6, 8, 12};
    int paletteC[4];
    for (int i = 0; i < 4; i++){
        int step = L / paletteK[i];
        paletteC[i] = rnd.next(0, step - 1);
    }

    int Z = 3 + (int)llround(f * 17.0);              // 3..20 zones
    int cellsPerZone = 5 + (int)llround(f * 195.0);   // 5..200 star cells / zone
    int linkDeg = 2 + (int)llround(f * 10.0);         // 2..12 adapter edges per side
    linkDeg = min(linkDeg, cellsPerZone);

    vector<int> zoneAngle(Z);
    vector<int> zoneCategory(Z, 0);
    zoneAngle[0] = rnd.next(0, L - 1);
    for (int z = 1; z < Z; z++){
        int cat = z % 4;
        zoneCategory[z] = cat;
        int delta;
        if (cat == 0)      delta = (rnd.next(0, 1) == 0) ? 8 : 16;             // gcd(.,24)=8
        else if (cat == 1) delta = (rnd.next(0, 1) == 0) ? 6 : 18;             // gcd(.,24)=6
        else if (cat == 2) delta = (rnd.next(0, 1) == 0) ? 4 : 20;             // gcd(.,24)=4
        else {
            static const int coprime[8] = {1,5,7,11,13,17,19,23};
            delta = coprime[rnd.next(0, 7)];                                    // gcd(.,24)=1
        }
        zoneAngle[z] = (zoneAngle[z - 1] + delta) % L;
    }

    // ---- lay out node ids: zones' star cells, then Z-1 adapters, then 3 padding stars
    vector<int> zoneStart(Z);
    int nid = 0;
    for (int z = 0; z < Z; z++){ zoneStart[z] = nid; nid += cellsPerZone; }
    int adapterStart = nid; nid += (Z - 1);
    int padStart = nid; nid += 3;
    int N = nid;

    vector<int> type(N, 0);
    for (int a = 0; a < Z - 1; a++) type[adapterStart + a] = 1;

    vector<int> cost(N, 0);
    for (int a = 0; a < Z - 1; a++) cost[adapterStart + a] = 1 + rnd.next(0, 3); // 1..4

    struct Edge { int u, v, a, len; };
    vector<Edge> edges;

    // intra-zone path edges (always matchable regardless of order chosen)
    for (int z = 0; z < Z; z++){
        for (int j = 0; j + 1 < cellsPerZone; j++){
            int u = zoneStart[z] + j, v = zoneStart[z] + j + 1;
            edges.push_back({u, v, zoneAngle[z], 10 + rnd.next(0, 40)});
        }
    }

    // adapter bridge edges: linkDeg edges into the tail of zone z, linkDeg into head of zone z+1.
    // Bridge edge weight scales with zone size so the adapter's payoff stays a
    // significant (not vanishing) share of the total strand length at every scale.
    int bridgeBase = 60 + cellsPerZone;
    for (int z = 0; z + 1 < Z; z++){
        int adapter = adapterStart + z;
        for (int j = 0; j < linkDeg; j++){
            int u = zoneStart[z] + (cellsPerZone - 1 - j);
            edges.push_back({adapter, u, zoneAngle[z], bridgeBase + rnd.next(0, 80)});
        }
        for (int j = 0; j < linkDeg; j++){
            int u = zoneStart[z + 1] + j;
            edges.push_back({adapter, u, zoneAngle[z + 1], bridgeBase + rnd.next(0, 80)});
        }
    }

    int M = (int)edges.size();

    printf("%d %d\n", N, M);
    for (int i = 0; i < N; i++) printf("%d%c", type[i], i + 1 < N ? ' ' : '\n');
    printf("4\n");
    for (int i = 0; i < 4; i++) printf("%d %d%c", paletteK[i], paletteC[i], i + 1 < 4 ? ' ' : '\n');
    int W1 = 6, W2 = 1, W3 = 3;
    printf("%d %d %d\n", W1, W2, W3);
    for (int i = 0; i < N; i++) printf("%d%c", cost[i], i + 1 < N ? ' ' : '\n');
    for (auto &e : edges) printf("%d %d %d %d\n", e.u, e.v, e.a, e.len);
    return 0;
}
