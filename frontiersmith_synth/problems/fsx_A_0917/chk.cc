#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Terrace Irrigation Retrofit"   family: matroid-intersection-packing
//
// Input:  V E K
//         cap_1 .. cap_K            (cap_1 >= E always: batch 1 = "standard stock", unlimited)
//         then E lines: u v w b     (candidate pipe segment: endpoints u,v in [1,V],
//                                    flow value w >= 1, material batch b in [1,K])
//
// Output: m, then m distinct pipe indices (1-indexed into the input list).
//
// Feasibility (BOTH must hold):
//   M1 (graphic matroid):    the chosen pipes, as edges over the V terraces, form a FOREST
//                             (no cycle) -- union-find check.
//   M2 (partition matroid):  for every batch b, the number of chosen pipes with that batch
//                             is <= cap_b.
// Objective (MAX): F = sum of w_i over chosen pipes.
//
// Baseline B (checker-computed): the maximum-weight spanning forest built using ONLY
//   batch-1 ("standard stock", never capacity-limited) pipes -- i.e. never touch a premium
//   batch at all. This is exactly what the `trivial` reference reproduces (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        return true;
    }
};

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int V = inf.readInt();
    int E = inf.readInt();
    int K = inf.readInt();
    vector<ll> cap(K + 1);
    for (int b = 1; b <= K; b++) cap[b] = inf.readLong();

    vector<int> eu(E + 1), ev(E + 1), eb(E + 1);
    vector<ll> ew(E + 1);
    for (int i = 1; i <= E; i++) {
        eu[i] = inf.readInt(1, V, "u");
        ev[i] = inf.readInt(1, V, "v");
        ew[i] = inf.readLong(1, (ll)1e9, "w");
        eb[i] = inf.readInt(1, K, "b");
    }

    // ---- internal baseline B: max-weight spanning forest using ONLY batch-1 pipes ----
    {
        vector<int> ids;
        for (int i = 1; i <= E; i++) if (eb[i] == 1) ids.push_back(i);
        sort(ids.begin(), ids.end(), [&](int a, int c) { return ew[a] > ew[c]; });
        DSU dsu(V + 1);
        ll B = 0;
        for (int id : ids) if (dsu.unite(eu[id], ev[id])) B += ew[id];
        if (B <= 0) B = 1;   // generator guarantees B > 0

        // ---- replay the participant's chosen pipe set ----
        int m = ouf.readInt(0, E, "m");
        vector<char> used(E + 1, 0);
        DSU dsu2(V + 1);
        vector<ll> cnt(K + 1, 0);
        ll F = 0;
        for (int j = 0; j < m; j++) {
            int id = ouf.readInt(1, E, "pipe_index");
            if (used[id]) quitf(_wa, "pipe %d selected more than once", id);
            used[id] = 1;
            if (!dsu2.unite(eu[id], ev[id]))
                quitf(_wa, "pipe %d closes a cycle among %d..%d already chosen", id, eu[id], ev[id]);
            cnt[eb[id]]++;
            if (cnt[eb[id]] > cap[eb[id]])
                quitf(_wa, "batch %d cap exceeded: %lld > %lld", eb[id], cnt[eb[id]], cap[eb[id]]);
            F += ew[id];
        }
        if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the pipe list");

        double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
        quitp(sc / 1000.0, "OK F=%lld B=%lld m=%d Ratio: %.6f", F, B, m, sc / 1000.0);
    }
    return 0;
}
