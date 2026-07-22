#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Star Lattice Interfaces" (family: zellij-star-lattice).
//
// Input:
//   N M
//   N ints type_i (0=star,1=adapter)
//   p ; p pairs (k C)            -- star palette order k -> canonical phase C
//   W1 W2 W3
//   N ints cost_i                -- adapter unit cost (0 for star cells)
//   M lines: u v a len           -- fixed edge angle a in [0,24), strand length len
//
// Output (participant): N lines "b k r"
//   b=0 -> cell not built (k,r ignored, but a token must still be present).
//   b=1 -> cell built with star/adapter order k and rotation r in [0,24).
//     STAR cell: k must be one of the p palette orders.
//     ADAPTER cell: k must be one of {3,4,6,8,12,24}.
//     Let step = 24 / k. For EVERY edge incident to this cell (angle a), require
//     a % step == r % step (edge-angle-matching) -- this must hold across ALL of
//     the cell's edges simultaneously (a single (k,r) must reconcile them all).
//
// Objective (MAX):
//   D       = number of distinct k used among built STAR cells.
//   SYM_SUM = sum over built star cells of bonus(k,r) = 10 if (r mod step)==C[k]
//             else 3   (canonical dihedral-alignment bonus).
//   LEN     = sum of len over edges with BOTH endpoints built.
//   COST    = sum over built ADAPTER cells of cost_i * k_i.
//   F = max(0, W1*D*SYM_SUM + W2*LEN - W3*COST).
//
// Baseline B (checker-computed "trivial" construction): build every star cell
//   with the SINGLE fixed order palette[0] (always self-consistent since every
//   star cell's own incident edges share one angle by construction); never
//   build any adapter. This is exactly the ladder's `trivial` reference.
//
// Score: sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static const int L = 24;
static const int ADAPTER_ALLOWED[6] = {3,4,6,8,12,24};
static ll g_B = 0;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    vector<int> type(N);
    for (int i = 0; i < N; i++) type[i] = inf.readInt(0, 1);
    int p = inf.readInt(1, 16);
    vector<int> paletteK(p), paletteC(p);
    map<int,int> kToC;
    set<int> paletteSet;
    for (int i = 0; i < p; i++){
        paletteK[i] = inf.readInt(1, L);
        paletteC[i] = inf.readInt(0, L - 1);
        kToC[paletteK[i]] = paletteC[i];
        paletteSet.insert(paletteK[i]);
    }
    ll W1 = inf.readLong();
    ll W2 = inf.readLong();
    ll W3 = inf.readLong();
    vector<int> cost(N);
    for (int i = 0; i < N; i++) cost[i] = inf.readInt(0, 1000000);

    vector<vector<int>> angles(N);           // incident edge angles per node
    vector<int> eu(M), ev(M), ea(M), elen(M);
    for (int i = 0; i < M; i++){
        int u = inf.readInt(0, N - 1);
        int v = inf.readInt(0, N - 1);
        int a = inf.readInt(0, L - 1);
        int len = inf.readInt(1, 1000000);
        eu[i]=u; ev[i]=v; ea[i]=a; elen[i]=len;
        angles[u].push_back(a);
        angles[v].push_back(a);
    }

    set<int> adapterAllowedSet(ADAPTER_ALLOWED, ADAPTER_ALLOWED + 6);

    // ---------------- internal baseline B: trivial construction ----------------
    {
        int k0 = paletteK[0];
        int step0 = L / k0;
        vector<char> builtTriv(N, 0);
        for (int i = 0; i < N; i++){
            if (type[i] != 0) continue; // trivial never builds adapters
            int rep = angles[i].empty() ? 0 : (angles[i][0] % step0);
            bool ok = true;
            for (int a : angles[i]) if (a % step0 != rep) { ok = false; break; }
            if (ok) builtTriv[i] = 1;
        }
        ll SYM_triv = 0; bool anyStar = false;
        for (int i = 0; i < N; i++){
            if (!builtTriv[i]) continue;
            anyStar = true;
            int rep = angles[i].empty() ? 0 : (angles[i][0] % step0);
            SYM_triv += (rep == paletteC[0]) ? 10 : 3;
        }
        ll D_triv = anyStar ? 1 : 0;
        ll LEN_triv = 0;
        for (int i = 0; i < M; i++) if (builtTriv[eu[i]] && builtTriv[ev[i]]) LEN_triv += elen[i];
        ll B = W1 * D_triv * SYM_triv + W2 * LEN_triv;
        if (B <= 0) B = 1;
        g_B = B;
    }

    // ---------------- read & validate participant output ----------------
    vector<char> built(N, 0);
    vector<int> K(N, 0), R(N, 0);
    for (int i = 0; i < N; i++){
        int b = ouf.readInt(0, 1, "build_flag");
        int k = ouf.readInt(0, L, "order");
        int r = ouf.readInt(0, 100000, "rotation");
        if (b == 0) continue;
        if (r < 0 || r >= L) quitf(_wa, "cell %d: rotation %d out of [0,%d)", i, r, L);
        if (type[i] == 0){
            if (!paletteSet.count(k)) quitf(_wa, "star cell %d: order %d not in palette", i, k);
        } else {
            if (!adapterAllowedSet.count(k)) quitf(_wa, "adapter cell %d: order %d not allowed", i, k);
        }
        int step = L / k;
        for (int a : angles[i]){
            if (a % step != r % step)
                quitf(_wa, "cell %d: edge angle %d does not match order %d rotation %d (step=%d)", i, a, k, r, step);
        }
        built[i] = 1; K[i] = k; R[i] = r;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---------------- objective ----------------
    set<int> usedOrders;
    ll SYM_SUM = 0;
    for (int i = 0; i < N; i++){
        if (!built[i] || type[i] != 0) continue;
        usedOrders.insert(K[i]);
        int step = L / K[i];
        int phase = R[i] % step;
        int C = kToC.count(K[i]) ? kToC[K[i]] : -1;
        SYM_SUM += (phase == C) ? 10 : 3;
    }
    ll D = (ll)usedOrders.size();

    ll LEN = 0;
    for (int i = 0; i < M; i++) if (built[eu[i]] && built[ev[i]]) LEN += elen[i];

    ll COST = 0;
    for (int i = 0; i < N; i++) if (built[i] && type[i] == 1) COST += (ll)cost[i] * K[i];

    ll F = W1 * D * SYM_SUM + W2 * LEN - W3 * COST;
    if (F < 0) F = 0;

    ll B = g_B;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld D=%lld SYM=%lld LEN=%lld COST=%lld Ratio: %.6f",
          F, B, D, SYM_SUM, LEN, COST, sc / 1000.0);
    return 0;
}
