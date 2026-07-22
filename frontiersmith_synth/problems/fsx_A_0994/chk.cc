#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Desert Temple of a Thousand Mirrors".
//
// Input:  P M K
//         P lines  x y q        (candidate site, id = 1-based line order)
//         K lines  dx dy e      (sun-step shadow offset + energy multiplier)
//
// Output: S  then S distinct site ids in [1,P], S<=M  -- the built mirror field.
//
// For a selected set S:
//   primitive spoke direction of site i: (x/g, y/g), g = gcd(|x|,|y|).
//   BLOCKED: site i is blocked (permanently, all day -> value 0) iff some
//     other selected site j shares i's spoke direction with strictly smaller g
//     (j stands exactly between the tower and i, on the same ray, so it
//     intercepts i's reflected beam toward the tower).
//   SHADED at step k: (unblocked) site i gets 0 for step k iff some other
//     selected site j sits EXACTLY at (x_i+dx_k, y_i+dy_k) -- i.e. exactly
//     "upsun" of i by that step's shadow-cast offset.
//   value(i) = 0 if blocked, else q_i * sum_{k: not shaded at k} e_k.
//   F = sum of value(i) over the selection.
//
// Baseline B (checker-computed, naive): the first min(M,P) candidates in
//   raster (generation) order -- an order unrelated to quality or geometry --
//   scored with the exact same rules. This is what solutions/trivial.cpp
//   reproduces (-> ratio ~0.1).
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

static ll igcd(ll a, ll b) {
    a = llabs(a); b = llabs(b);
    while (b) { ll t = a % b; a = b; b = t; }
    return a;
}

int P, M, K;
vector<ll> X, Y, Q, DX, DY, E;

static inline ll encPos(ll x, ll y) {
    return (x + 2000000000LL) * 4000000001LL + (y + 2000000000LL);
}

ll computeF(const vector<int>& sel) {
    int S = (int)sel.size();
    unordered_map<ll, int> posIndex;
    posIndex.reserve((size_t)S * 2 + 1);
    for (int idx = 0; idx < S; idx++) {
        int id = sel[idx];
        posIndex[encPos(X[id], Y[id])] = id;
    }
    map<pair<ll, ll>, ll> groupMin;
    vector<ll> gval(S);
    vector<pair<ll, ll>> pd(S);
    for (int idx = 0; idx < S; idx++) {
        int id = sel[idx];
        ll g = igcd(X[id], Y[id]);
        pair<ll, ll> key = {X[id] / g, Y[id] / g};
        gval[idx] = g;
        pd[idx] = key;
        auto it = groupMin.find(key);
        if (it == groupMin.end() || g < it->second) groupMin[key] = g;
    }
    ll F = 0;
    for (int idx = 0; idx < S; idx++) {
        int id = sel[idx];
        if (gval[idx] > groupMin[pd[idx]]) continue;   // blocked
        ll val = 0;
        for (int k = 1; k <= K; k++) {
            ll tx = X[id] + DX[k], ty = Y[id] + DY[k];
            auto it = posIndex.find(encPos(tx, ty));
            if (it != posIndex.end() && it->second != id) continue;  // shaded this step
            val += E[k];
        }
        F += Q[id] * val;
    }
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    P = inf.readInt();
    M = inf.readInt();
    K = inf.readInt();
    X.assign(P + 1, 0); Y.assign(P + 1, 0); Q.assign(P + 1, 0);
    for (int i = 1; i <= P; i++) {
        X[i] = inf.readLong();
        Y[i] = inf.readLong();
        Q[i] = inf.readLong();
    }
    DX.assign(K + 1, 0); DY.assign(K + 1, 0); E.assign(K + 1, 0);
    for (int k = 1; k <= K; k++) {
        DX[k] = inf.readLong();
        DY[k] = inf.readLong();
        E[k] = inf.readLong();
    }

    // ---- internal baseline B: naive first-M-in-input-order selection,
    // ignoring quality and geometry entirely. ----
    vector<int> baseSel;
    int mb = min(M, P);
    baseSel.reserve(mb);
    for (int i = 1; i <= mb; i++) baseSel.push_back(i);
    ll B = computeF(baseSel);
    if (B <= 0) B = 1;

    // ---- read participant selection ----
    int S = ouf.readInt(0, M, "num_selected");
    vector<int> sel(S);
    vector<char> seen(P + 1, 0);
    for (int i = 0; i < S; i++) {
        int id = ouf.readInt(1, P, "site_id");
        if (seen[id]) quitf(_wa, "site %d selected more than once", id);
        seen[id] = 1;
        sel[i] = id;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after selection list");

    ll F = computeF(sel);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld S=%d Ratio: %.6f", F, B, S, sc / 1000.0);
    return 0;
}
