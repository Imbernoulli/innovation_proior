#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Checker / scorer for "Quarry Ignition".
//
// Input:  N M ; N charges (x y v r t c).
// Output: M distinct integers in [1,N], each referring to an ignitable (c=1)
//         charge -- the ignition set S.
//
// Cascade (deterministic, order-independent): charges in S detonate at round 0.
// A not-yet-detonated charge j detonates once the number of DISTINCT already-
// detonated charges i with dist(i,j) <= r_i is >= t_j. Repeat to a fixed point.
// F = total value of the closure. Baseline B = closure value of the M ignitable
// charges with the largest v_i (ties -> smaller index) -- the value-only pick.
// Score (max): sc = min(1000, 100*F/max(1,B)); ratio = sc/1000.
// -----------------------------------------------------------------------------

int N, M;
vector<ll> X, Y, V, R, T;
vector<int> C;
vector<vector<int>> outAdj;   // outAdj[i] = list of j with edge i->j (i's blast reaches j)

static inline bool reaches(int i, int j){
    ll dx = X[i] - X[j], dy = Y[i] - Y[j];
    ll d2 = dx * dx + dy * dy;
    ll r2 = R[i] * R[i];
    return d2 <= r2;
}

void buildGraph(){
    outAdj.assign(N, {});
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if (i != j && reaches(i, j))
                outAdj[i].push_back(j);
}

// Deterministic threshold cascade from an ignition set (list of 0-indexed charges).
// Returns total value of the closure.
ll simulate(const vector<int>& ignite){
    vector<char> det(N, 0);
    vector<int> cnt(N, 0);
    queue<int> q;
    for (int i : ignite){
        if (!det[i]){ det[i] = 1; q.push(i); }
    }
    while (!q.empty()){
        int u = q.front(); q.pop();
        for (int j : outAdj[u]){
            if (det[j]) continue;
            cnt[j]++;
            if (cnt[j] >= T[j]){
                det[j] = 1;
                q.push(j);
            }
        }
    }
    ll total = 0;
    for (int i = 0; i < N; i++) if (det[i]) total += V[i];
    return total;
}

int main(int argc, char* argv[]){
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    X.assign(N, 0); Y.assign(N, 0); V.assign(N, 0); R.assign(N, 0); T.assign(N, 0);
    C.assign(N, 0);
    for (int i = 0; i < N; i++){
        X[i] = inf.readLong();
        Y[i] = inf.readLong();
        V[i] = inf.readLong();
        R[i] = inf.readLong();
        T[i] = inf.readLong();
        C[i] = (int)inf.readLong();
    }
    buildGraph();

    // ---- read participant ignition set ----
    vector<int> chosen; chosen.reserve(M);
    vector<char> seen(N + 1, 0);
    for (int k = 0; k < M; k++){
        int idx = ouf.readInt(1, N, "idx");   // bounded: rejects garbage/out-of-range/nan
        if (seen[idx]) quitf(_wa, "duplicate ignition index %d", idx);
        seen[idx] = 1;
        int zi = idx - 1;
        if (C[zi] != 1) quitf(_wa, "charge %d is not ignitable (shielded)", idx);
        chosen.push_back(zi);
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    ll F = simulate(chosen);

    // ---- baseline: the M ignitable charges with the largest value ----
    vector<int> ignitable;
    for (int i = 0; i < N; i++) if (C[i] == 1) ignitable.push_back(i);
    sort(ignitable.begin(), ignitable.end(), [&](int a, int b){
        if (V[a] != V[b]) return V[a] > V[b];
        return a < b;
    });
    vector<int> baseline(ignitable.begin(), ignitable.begin() + M);
    ll B = simulate(baseline);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
