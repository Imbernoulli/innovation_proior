#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Binding the Atlas".  family: biased-fanout-search-tree.
// OBJECTIVE (MIN):
//   L(P)=ceil(f_P/B) = line count of a page holding f_P tiles.
//   lookup(x)  = sum of L(P) over pages P on the root->page(x) path ; weighted by w_x.
//   scan[lo,hi]= sum of L(P) over the DISTINCT pages that store some tile in [lo,hi] ;
//                weighted by c.
//   F = sum_x w_x*lookup(x)  +  sum_scans c*scan(lo,hi).
// BASELINE Bref (checker-built, blind to B/demand/scans): the balanced BINARY search
//   tree over 1..N (one tile per page, every L=1). lookup(x)=binary depth ; a scan of
//   R tiles touches R singleton pages. The `trivial` reference reproduces this tree
//   exactly -> ratio 0.1.
// Score: sc = min(1000, 100*Bref/max(1,F)); ratio=sc/1000 in [0,1] (cap = 10x Bref).
// -----------------------------------------------------------------------------

int N, B, S;
vector<ll> w;
vector<int> sLo, sHi;
vector<ll> sC;

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    B = inf.readInt();
    S = inf.readInt();
    w.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) w[i] = inf.readLong();
    sLo.resize(S); sHi.resize(S); sC.resize(S);
    for (int i = 0; i < S; i++){
        sLo[i] = inf.readInt();
        sHi[i] = inf.readInt();
        sC[i]  = inf.readLong();
    }

    // ---- read participant tree ----
    int M = ouf.readInt(1, N, "M");
    int root = ouf.readInt(1, M, "root");
    vector<int> nf(M + 1);
    vector<vector<int>> keys(M + 1), ch(M + 1);
    for (int i = 1; i <= M; i++){
        int f = ouf.readInt(1, N, "f");
        nf[i] = f;
        keys[i].resize(f);
        ch[i].resize(f + 1);
        int prev = 0;
        for (int j = 0; j < f; j++){
            int k = ouf.readInt(1, N, "key");
            if (k <= prev) quitf(_wa, "page %d keys not strictly increasing", i);
            prev = k; keys[i][j] = k;
        }
        for (int j = 0; j <= f; j++) ch[i][j] = ouf.readInt(0, M, "child");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the tree");

    // line count per page
    vector<ll> lc(M + 1);
    for (int i = 1; i <= M; i++) lc[i] = (nf[i] + B - 1) / B;

    // ---- validate via iterative in-order; record page(x) & root->page path cost ----
    vector<int> pageOf(N + 1, 0);
    vector<ll> pathCost(M + 1, 0);       // sum L over root..node
    vector<char> vis(M + 1, 0);
    struct Fr { int nd, pos; };
    vector<Fr> st;
    vis[root] = 1; pathCost[root] = lc[root];
    st.push_back({root, 0});
    int visited = 1;
    ll expected = 1;
    while (!st.empty()){
        Fr cur = st.back(); st.pop_back();
        int nd = cur.nd, f = nf[nd];
        if (cur.pos > 2 * f) continue;
        st.push_back({nd, cur.pos + 1});
        int p = cur.pos;
        if (p % 2 == 0){
            int c = ch[nd][p / 2];
            if (c != 0){
                if (vis[c]) quitf(_wa, "page %d reused (output is not a tree)", c);
                vis[c] = 1; visited++;
                pathCost[c] = pathCost[nd] + lc[c];
                st.push_back({c, 0});
            }
        } else {
            int k = keys[nd][(p - 1) / 2];
            if (k != expected) quitf(_wa, "in-order walk expected tile %lld but read %d", expected, k);
            pageOf[k] = nd;
            expected++;
        }
    }
    if (expected != (ll)N + 1)
        quitf(_wa, "tree does not hold all tiles 1..%d (in-order reached %lld)", N, expected - 1);
    if (visited != M)
        quitf(_wa, "only %d of %d pages are reachable from the root", visited, M);

    // ---- participant objective F ----
    ll F = 0;
    for (int x = 1; x <= N; x++) if (w[x] > 0) F += w[x] * pathCost[pageOf[x]];
    vector<int> stamp(M + 1, 0);
    int cs = 0;
    for (int i = 0; i < S; i++){
        int lo = sLo[i], hi = sHi[i];
        ++cs; ll cost = 0;
        for (int y = lo; y <= hi; y++){
            int nd = pageOf[y];
            if (stamp[nd] != cs){ stamp[nd] = cs; cost += lc[nd]; }
        }
        F += sC[i] * cost;
    }
    if (F <= 0) F = 1;

    // ---- baseline Bref: balanced binary tree over 1..N (each L=1) ----
    vector<int> bdepth(N + 1, 0);
    {
        vector<array<int,3>> stk;
        stk.push_back({1, N, 1});
        while (!stk.empty()){
            auto a = stk.back(); stk.pop_back();
            int lo = a[0], hi = a[1], d = a[2];
            if (lo > hi) continue;
            int mid = (lo + hi) / 2;
            bdepth[mid] = d;
            stk.push_back({lo, mid - 1, d + 1});
            stk.push_back({mid + 1, hi, d + 1});
        }
    }
    ll Bref = 0;
    for (int x = 1; x <= N; x++) if (w[x] > 0) Bref += w[x] * (ll)bdepth[x];
    for (int i = 0; i < S; i++) Bref += sC[i] * (ll)(sHi[i] - sLo[i] + 1);
    if (Bref <= 0) Bref = 1;

    double sc = min(1000.0, 100.0 * (double)Bref / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld pages=%d Ratio: %.6f", F, Bref, M, sc / 1000.0);
    return 0;
}
