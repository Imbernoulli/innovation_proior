#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Fragment-Based Ligand Assembly".
//
// Input:  M C Wmax Lam Rb ; then M lines  w b val p ; then C lines of C ints (q).
// Output: S ; S distinct fragment indices ; E ; E bond pairs (a b).
//
// Feasibility: distinct in-range indices; bonds between selected fragments,
// a!=b; no parallel bonds; degree(i) <= val_i; the selected fragments + bonds
// form ONE connected component.
//
// Objective (maximize):
//   F = sum b_i (selected)
//     + sum q[p_a][p_b] over bonds
//     + Rb * (E - S + 1)                         # independent rings (connected => >=0)
//     - Lam * max(0, W - Wmax),  W = sum w_i (selected).
//
// Baseline B = max_i ( b_i - Lam*max(0, w_i - Wmax) )  > 0 (generator guarantee).
// Score (max): sc = clamp(100 * F / max(1,B), 0, 1000); ratio = sc/1000.
//   best single fragment -> ratio 0.1 ; a good assembly -> higher, capped 1.0.
// -----------------------------------------------------------------------------

int M, C;
ll Wmax, Lam, Rb;
vector<ll> w, b, val, p;
vector<vector<ll>> q;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    M   = inf.readInt();
    C   = inf.readInt();
    Wmax= inf.readLong();
    Lam = inf.readLong();
    Rb  = inf.readLong();

    w.assign(M + 1, 0); b.assign(M + 1, 0); val.assign(M + 1, 0); p.assign(M + 1, 0);
    for (int i = 1; i <= M; i++){
        w[i]   = inf.readLong();
        b[i]   = inf.readLong();
        val[i] = inf.readLong();
        p[i]   = inf.readLong();
    }
    q.assign(C, vector<ll>(C, 0));
    for (int a = 0; a < C; a++)
        for (int c = 0; c < C; c++)
            q[a][c] = inf.readLong();

    // ---- baseline: best single fragment ----
    ll B = LLONG_MIN;
    for (int i = 1; i <= M; i++){
        ll over = max((ll)0, w[i] - Wmax);
        ll v = b[i] - Lam * over;
        B = max(B, v);
    }
    if (B <= 0) B = 1;  // safety; generator guarantees a positive single fragment

    // ---- read participant output ----
    int S = ouf.readInt(1, M, "S");
    vector<int> sel(S);
    vector<char> chosen(M + 1, 0);
    for (int i = 0; i < S; i++){
        int idx = ouf.readInt(1, M, "fragment_index");
        if (chosen[idx]) quitf(_wa, "fragment %d selected more than once", idx);
        chosen[idx] = 1;
        sel[i] = idx;
    }

    // map fragment id -> local position (0..S-1) for union-find / degree
    unordered_map<int,int> pos;
    pos.reserve(S * 2);
    for (int i = 0; i < S; i++) pos[sel[i]] = i;

    int maxBonds = 2 * M + 5;  // total valence <= 4M => at most 2M bonds
    int E = ouf.readInt(0, maxBonds, "E");

    vector<int> deg(S, 0);
    vector<int> par(S);
    for (int i = 0; i < S; i++) par[i] = i;
    function<int(int)> find = [&](int x){ while (par[x] != x){ par[x] = par[par[x]]; x = par[x]; } return x; };

    set<pair<int,int>> bondSet;
    ll bondScore = 0;
    for (int e = 0; e < E; e++){
        int a = ouf.readInt(1, M, "bond_a");
        int c = ouf.readInt(1, M, "bond_b");
        if (a == c) quitf(_wa, "self-bond on fragment %d", a);
        if (!chosen[a]) quitf(_wa, "bond endpoint %d is not a selected fragment", a);
        if (!chosen[c]) quitf(_wa, "bond endpoint %d is not a selected fragment", c);
        int lo = min(a, c), hi = max(a, c);
        if (bondSet.count({lo, hi})) quitf(_wa, "parallel bond between %d and %d", lo, hi);
        bondSet.insert({lo, hi});
        int pa = pos[a], pc = pos[c];
        deg[pa]++; deg[pc]++;
        if (deg[pa] > val[a]) quitf(_wa, "fragment %d exceeds valence %lld", a, val[a]);
        if (deg[pc] > val[c]) quitf(_wa, "fragment %d exceeds valence %lld", c, val[c]);
        int ra = find(pa), rc = find(pc);
        if (ra != rc) par[ra] = rc;
        bondScore += q[ p[a] ][ p[c] ];
    }

    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // ---- connectivity ----
    int roots = 0;
    for (int i = 0; i < S; i++) if (find(i) == i) roots++;
    if (roots != 1) quitf(_wa, "assembled molecule is not connected (%d components)", roots);

    // ---- objective ----
    ll fragScore = 0, W = 0;
    for (int i = 0; i < S; i++){ fragScore += b[sel[i]]; W += w[sel[i]]; }
    ll rings = (ll)E - (ll)S + 1;                  // >= 0 since connected
    ll ringScore = Rb * rings;
    ll over = max((ll)0, W - Wmax);
    ll weightPen = Lam * over;

    ll F = fragScore + bondScore + ringScore - weightPen;

    double sc = 100.0 * (double)F / (double)max((ll)1, B);
    if (sc < 0.0) sc = 0.0;
    if (sc > 1000.0) sc = 1000.0;

    quitp(sc / 1000.0, "OK F=%lld B=%lld S=%d E=%d rings=%lld Ratio: %.6f",
          F, B, S, E, rings, sc / 1000.0);
    return 0;
}
