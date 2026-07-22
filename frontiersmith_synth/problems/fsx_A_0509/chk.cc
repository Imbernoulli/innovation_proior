#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Sawmill with one dying thin blade".
//
// Input:  W H T t L D ; then D lines  w h v  (demanded rectangle sizes + values).
// Output: a guillotine cut tree in PRE-ORDER over the sheet rect (W x H):
//    leaf : "0 d"       d=0 waste, or d in [1,D] claim demand d (rect must equal w x h).
//    cut  : "1 o b a"   o=0 vertical / o=1 horizontal ; b=0 thick / b=1 thin ;
//                       a = size of the FIRST child along the cut axis; then the two
//                       children follow in pre-order. kerf = (b==0?T:t).
//           vertical  on (w,h): child1=(a,h)          child2=(w-kerf-a, h)
//           horizontal on (w,h): child1=(w,a)          child2=(w, h-kerf-a)
//           require a>=1 and (dim-kerf-a)>=1. A thin cut consumes its length
//           (vertical: h, horizontal: w) from budget L.
//
// Objective (MAX): F = sum of values of claimed demands, subject to total thin
//   cut-length <= L and each demand claimed at most once.
//
// Baseline B (checker-computed, thick-only shelf packing): group demands by equal
//   height; in each group place pieces widest-first with the THICK blade
//   (reserve kerf T per placed piece). B = sum of placed values. This is exactly
//   the "cut every base piece with the reliable blade, ignore the dying one"
//   construction, and it is what the trivial reference reproduces (-> ratio 0.1).
// Score (max): sc = min(1000, 100 * F / max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    ll W = inf.readLong();
    ll H = inf.readLong();
    ll T = inf.readLong();
    ll t = inf.readLong();
    ll L = inf.readLong();
    int D = inf.readInt();
    vector<ll> dw(D + 1), dh(D + 1), dv(D + 1);
    for (int i = 1; i <= D; i++){
        dw[i] = inf.readLong();
        dh[i] = inf.readLong();
        dv[i] = inf.readLong();
    }

    // ---- internal baseline B: thick-only shelf packing (group by height) ----
    map<ll, vector<int>> groups;
    for (int i = 1; i <= D; i++) groups[dh[i]].push_back(i);
    ll B = 0;
    for (auto &kv : groups){
        vector<int> ids = kv.second;
        sort(ids.begin(), ids.end(), [&](int a, int b){ return dw[a] > dw[b]; });
        ll rem = W;
        for (int id : ids){
            if (dw[id] + T <= rem){ B += dv[id]; rem -= dw[id] + T; }
        }
    }
    if (B <= 0) B = 1;   // generator guarantees B>0

    // ---- replay the participant's guillotine tree (iterative pre-order) ----
    vector<char> claimed(D + 1, 0);
    ll F = 0, thinUsed = 0, nodes = 0;
    const ll NODECAP = 3000000;
    vector<pair<ll,ll>> st;
    st.push_back({W, H});
    while (!st.empty()){
        if (++nodes > NODECAP) quitf(_wa, "tree too large (> %lld nodes)", NODECAP);
        pair<ll,ll> pr = st.back(); st.pop_back();
        ll w = pr.first, h = pr.second;
        int typ = ouf.readInt(0, 1, "node_type");
        if (typ == 0){                                   // leaf
            int d = ouf.readInt(0, D, "claim");
            if (d >= 1){
                if (claimed[d]) quitf(_wa, "demand %d claimed more than once", d);
                if (!(dw[d] == w && dh[d] == h))
                    quitf(_wa, "leaf %lldx%lld does not equal demand %d (%lldx%lld)",
                          w, h, d, dw[d], dh[d]);
                claimed[d] = 1;
                F += dv[d];
            }
        } else {                                         // cut
            int o = ouf.readInt(0, 1, "orient");
            int b = ouf.readInt(0, 1, "blade");
            ll kerf = (b == 0 ? T : t);
            ll dim  = (o == 0 ? w : h);
            if (dim < kerf + 2) quitf(_wa, "rect %lldx%lld too small for a cut", w, h);
            ll a = ouf.readLong(1LL, dim - kerf - 1, "a");
            ll rem = dim - kerf - a;                     // >= 1 by the bound
            if (b == 1) thinUsed += (o == 0 ? h : w);    // thin cut-length
            if (o == 0){ st.push_back({rem, h}); st.push_back({a, h}); }
            else       { st.push_back({w, rem}); st.push_back({w, a}); }
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after the tree");
    if (thinUsed > L)
        quitf(_wa, "thin-blade wear budget exceeded: used %lld > L=%lld", thinUsed, L);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld thin=%lld/%lld Ratio: %.6f",
          F, B, thinUsed, L, sc / 1000.0);
    return 0;
}
