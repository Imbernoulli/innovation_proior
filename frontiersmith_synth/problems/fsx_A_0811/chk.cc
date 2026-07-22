#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Valve Segmentation Under a Double-Burst Sweep".
//
// Input:  N M V C ; c_1..c_C (candidate/valve-eligible pipe indices) ;
//         M lines u_k v_k d_k ; L ; L lines i_t j_t (double-burst pairs).
// Output: k, then k distinct pipe indices (all must be in the candidate set).
//
// Baseline B (checker-computed, do-nothing): install zero valves -> the whole
//   tree is one segment, B = total demand of every pipe. This is exactly what
//   the trivial reference reproduces (-> ratio 0.1).
// Objective F (MIN): after removing the chosen valve pipes, every remaining
//   pipe belongs to a segment (a connected component); F = max over every
//   un-valved pipe of its segment's total demand, and over every double-burst
//   pair of loss(i,j) = segDemand(seg(i)) if same segment else the SUM of both
//   segments' demand.
// Score (min): sc = min(1000, 100 * B / max(1,F)); ratio = sc/1000.
// -----------------------------------------------------------------------------

struct DSU {
    vector<int> p;
    DSU(int n) : p(n + 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void uni(int a, int b) { a = find(a); b = find(b); if (a != b) p[a] = b; }
};

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int N = inf.readInt();
    int M = inf.readInt();
    int V = inf.readInt();
    int C = inf.readInt();

    vector<char> isCand(M + 1, 0);
    for (int i = 0; i < C; i++) {
        int c = inf.readInt(1, M, "candidate");
        isCand[c] = 1;
    }

    vector<int> eu(M + 1), ev(M + 1);
    vector<ll> ed(M + 1);
    ll totalDemand = 0;
    for (int k = 1; k <= M; k++) {
        eu[k] = inf.readInt(1, N, "u");
        ev[k] = inf.readInt(1, N, "v");
        ed[k] = inf.readLong(1, (ll)1e9, "d");
        totalDemand += ed[k];
    }

    int L = inf.readInt();
    vector<int> pi(L), pj(L);
    for (int t = 0; t < L; t++) {
        pi[t] = inf.readInt(1, M, "pi");
        pj[t] = inf.readInt(1, M, "pj");
    }

    ll B = max((ll)1, totalDemand);

    // ---- read participant output ----
    int k = ouf.readInt(0, V, "k");
    vector<int> chosen(k);
    vector<char> isChosen(M + 1, 0);
    for (int t = 0; t < k; t++) {
        int r = ouf.readInt(1, M, "valve_idx");
        if (isChosen[r]) quitf(_wa, "pipe %d chosen as a valve more than once", r);
        if (!isCand[r]) quitf(_wa, "pipe %d is not a valve-eligible candidate", r);
        isChosen[r] = 1;
        chosen[t] = r;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- segments: union all NON-chosen pipes ----
    DSU dsu(N);
    for (int e = 1; e <= M; e++)
        if (!isChosen[e]) dsu.uni(eu[e], ev[e]);

    vector<ll> segDemand(N + 1, 0);
    for (int e = 1; e <= M; e++) {
        if (isChosen[e]) continue;
        int r = dsu.find(eu[e]);
        segDemand[r] += ed[e];
    }

    ll F1 = 0;
    for (int e = 1; e <= M; e++) {
        if (isChosen[e]) continue;
        int r = dsu.find(eu[e]);
        F1 = max(F1, segDemand[r]);
    }

    ll F2 = 0;
    for (int t = 0; t < L; t++) {
        // pi[t], pj[t] are guaranteed (by the generator) to never be candidates,
        // hence never chosen as a valve -- always resolvable to a segment.
        int ri = dsu.find(eu[pi[t]]);
        int rj = dsu.find(eu[pj[t]]);
        ll loss = (ri == rj) ? segDemand[ri] : (segDemand[ri] + segDemand[rj]);
        F2 = max(F2, loss);
    }

    ll F = max(F1, F2);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld k=%d Ratio: %.6f", F, B, k, sc / 1000.0);
    return 0;
}
