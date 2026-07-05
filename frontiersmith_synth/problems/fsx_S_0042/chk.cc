#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, T;
vector<ll> C;                       // telescope budgets, 1-indexed size T+1
vector<vector<ll>> cost, val;       // [i in 0..N-1][j in 0..T-1]

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    T = inf.readInt();
    C.assign(T + 1, 0);
    for (int j = 1; j <= T; j++) C[j] = inf.readInt();

    cost.assign(N, vector<ll>(T));
    val.assign(N, vector<ll>(T));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < T; j++) {
            cost[i][j] = inf.readInt();
            val[i][j]  = inf.readInt();
        }

    // ---- internal baseline B: deterministic first-fit reference schedule ----
    vector<ll> remB(T, 0);
    for (int j = 0; j < T; j++) remB[j] = C[j + 1];
    ll B = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < T; j++) {
            if (cost[i][j] <= remB[j]) {
                remB[j] -= cost[i][j];
                B += val[i][j];
                break;
            }
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant's schedule ----
    vector<ll> load(T + 1, 0);
    ll F = 0;
    for (int i = 0; i < N; i++) {
        int a = ouf.readInt(0, T, "assign");
        if (a > 0) {
            load[a] += cost[i][a - 1];
            F += val[i][a - 1];
        }
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    for (int j = 1; j <= T; j++) {
        if (load[j] > C[j])
            quitf(_wa, "telescope %d over budget: used %lld > budget %lld", j, load[j], C[j]);
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
