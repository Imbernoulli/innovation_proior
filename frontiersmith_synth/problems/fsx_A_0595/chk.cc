#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Order Slab"  (family: guillotine-cut-tree-planner)
//
// Input:  W H K D ; then D lines  w h v c  (demand type: size, per-piece value, cap).
// Output: a guillotine cut tree in PRE-ORDER over the sheet rect (W x H):
//    leaf : "0 t"      t=0 scrap, or t in [1,D] claims ONE unit of demand t. The
//                      leaf rect must equal (w_t, h_t) EXACTLY.
//    cut  : "1 o a"    o=0 vertical (split width), o=1 horizontal (split height);
//                      a = size of the FIRST child along the cut axis; the two
//                      children then follow in pre-order.
//           vertical  on (w,h): child1=(a,h)   child2=(w-a,h)
//           horizontal on (w,h): child1=(w,a)   child2=(w,h-a)
//           require 1<=a and (dim-a)>=1. At most K cut nodes total.
//
// Objective (MAX): F = sum over t of v_t * min(c_t, claimed_t).  Claiming a type
//   more than c_t times is legal but the extra claims earn nothing.
//
// Baseline B (checker-computed): the best SINGLE SHELF -- one band holding as many
//   copies of one type as fit across the width, capped by c_t:
//       B = max_t  v_t * min(c_t, floor(W / w_t))   over types with w_t<=W, h_t<=H.
//   This is exactly what the trivial reference builds (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int D;
ll K;
vector<ll> dw, dh, dv, dc;
vector<ll> claimed;
ll cutCount = 0;

// iterative pre-order replay with an explicit rectangle stack
void replay(ll W, ll H){
    vector<pair<ll,ll>> st;
    st.push_back({W, H});
    while (!st.empty()){
        pair<ll,ll> pr = st.back(); st.pop_back();
        ll w = pr.first, h = pr.second;
        int typ = ouf.readInt(0, 1, "node_type");
        if (typ == 0){                                  // leaf
            int t = ouf.readInt(0, D, "claim");
            if (t >= 1){
                if (!(dw[t] == w && dh[t] == h))
                    quitf(_wa, "leaf %lldx%lld does not equal demand %d (%lldx%lld)",
                          w, h, t, dw[t], dh[t]);
                claimed[t]++;
            }
        } else {                                        // cut
            if (++cutCount > K) quitf(_wa, "used more than K=%lld cuts", K);
            int o = ouf.readInt(0, 1, "orient");
            if (o == 0){                                // vertical: split width
                if (w < 2) quitf(_wa, "cannot vertical-cut width %lld", w);
                ll a = ouf.readInt(1, w - 1, "a");
                // push children so child1 is processed first (pre-order): push child2 then child1
                st.push_back({w - a, h});
                st.push_back({a, h});
            } else {                                    // horizontal: split height
                if (h < 2) quitf(_wa, "cannot horizontal-cut height %lld", h);
                ll a = ouf.readInt(1, h - 1, "a");
                st.push_back({w, h - a});
                st.push_back({w, a});
            }
        }
    }
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll W = inf.readLong();
    ll H = inf.readLong();
    K   = inf.readLong();
    D   = inf.readInt();
    dw.assign(D + 1, 0); dh.assign(D + 1, 0); dv.assign(D + 1, 0); dc.assign(D + 1, 0);
    for (int i = 1; i <= D; i++){
        dw[i] = inf.readLong();
        dh[i] = inf.readLong();
        dv[i] = inf.readLong();
        dc[i] = inf.readLong();
    }

    // ---- internal baseline B: best single shelf ----
    ll B = 0;
    for (int i = 1; i <= D; i++){
        if (dw[i] <= W && dh[i] <= H){
            ll fit = min(dc[i], W / dw[i]);
            B = max(B, dv[i] * fit);
        }
    }
    if (B <= 0) B = 1;   // generator guarantees a feasible type

    // ---- replay participant tree ----
    claimed.assign(D + 1, 0);
    replay(W, H);
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the cut tree");

    // ---- objective ----
    ll F = 0;
    for (int t = 1; t <= D; t++)
        F += dv[t] * min(dc[t], claimed[t]);

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
