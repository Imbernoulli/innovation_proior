#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Dispersed Relay Coverage Under a Roaming Jammer".
//
// Input:  N K M L r R ; N demand points (x y w t) ; M jammer positions (gx gy).
// Output: exactly K towers (x y), 0<=x,y<=L.
//
// Objective (MAX): for jammer g, tower j survives iff d2(tower_j,g) > R^2.
// Demand i is reached by surviving tower j iff d2(tower_j,point_i) <= r^2.
// C(g) = sum_i w_i * min(cnt_i(g), t_i) / t_i .   F = min over g of C(g).
//
// Baseline B (checker-computed): a fixed uniform grid over [0,L]x[0,L] sized only
// from K (ignores weights/targets/jammers) scored with the same F -- this is what
// the trivial reference reproduces -> ratio 0.1.
// Score (max): sc = min(1000, 100 * F / max(eps,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N, K, M, L, r, R;
vector<ll> px, py, pw; vector<int> pt;
vector<ll> gx, gy;

static inline ll d2(ll ax, ll ay, ll bx, ll by){
    ll dx = ax - bx, dy = ay - by;
    return dx * dx + dy * dy;
}

double scoreConfig(const vector<pair<ll,ll>>& towers){
    ll r2 = (ll)r * r, R2 = (ll)R * R;
    int T = (int)towers.size();
    vector<char> survive(T);
    double F = -1.0;
    for (int g = 0; g < M; g++){
        for (int j = 0; j < T; j++)
            survive[j] = d2(towers[j].first, towers[j].second, gx[g], gy[g]) > R2 ? 1 : 0;
        double Cg = 0.0;
        for (int i = 0; i < N; i++){
            int cnt = 0;
            for (int j = 0; j < T; j++){
                if (!survive[j]) continue;
                if (d2(towers[j].first, towers[j].second, px[i], py[i]) <= r2) cnt++;
            }
            if (cnt > pt[i]) cnt = pt[i];
            Cg += (double)pw[i] * (double)cnt / (double)pt[i];
        }
        if (F < 0.0 || Cg < F) F = Cg;
    }
    if (F < 0.0) F = 0.0;
    return F;
}

// fixed uniform grid layout over [0,L]x[0,L], derived ONLY from K and L.
vector<pair<ll,ll>> gridTowers(int K_, int L_){
    int s = (int)ceil(sqrt((double)K_));
    if (s < 1) s = 1;
    vector<pair<ll,ll>> t;
    for (int idx = 0; idx < K_; idx++){
        int row = idx / s, col = idx % s;
        ll x = (ll)(2 * col + 1) * L_ / (2LL * s);
        ll y = (ll)(2 * row + 1) * L_ / (2LL * s);
        t.push_back({x, y});
    }
    return t;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt(); K = inf.readInt(); M = inf.readInt();
    L = inf.readInt(); r = inf.readInt(); R = inf.readInt();
    px.resize(N); py.resize(N); pw.resize(N); pt.resize(N);
    for (int i = 0; i < N; i++){
        px[i] = inf.readInt(); py[i] = inf.readInt();
        pw[i] = inf.readInt(); pt[i] = inf.readInt();
    }
    gx.resize(M); gy.resize(M);
    for (int g = 0; g < M; g++){ gx[g] = inf.readInt(); gy[g] = inf.readInt(); }

    // ---- baseline B: location/weight/jammer-agnostic uniform grid ----
    double B = scoreConfig(gridTowers(K, L));
    if (B < 1e-6) B = 1e-6;

    // ---- read participant towers (strict feasibility) ----
    vector<pair<ll,ll>> towers(K);
    for (int k = 0; k < K; k++){
        int x = ouf.readInt(0, L, "x");
        int y = ouf.readInt(0, L, "y");
        towers[k] = {x, y};
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly K=%d lines)", K);

    double F = scoreConfig(towers);
    double sc = min(1000.0, 100.0 * F / max(1e-6, B));
    quitp(sc / 1000.0, "OK F=%.6f B=%.6f Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
