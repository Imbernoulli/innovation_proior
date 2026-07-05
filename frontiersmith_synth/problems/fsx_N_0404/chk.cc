#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int N, M, d;
vector<ll> p;               // 1-indexed runtimes
vector<vector<ll>> X;       // 1-indexed profiles

static inline ll setup2(int a, int b) {
    ll s = 0;
    for (int k = 0; k < d; k++) { ll dd = X[a][k] - X[b][k]; s += dd * dd; }
    return s;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    N = inf.readInt();
    M = inf.readInt();
    d = inf.readInt();
    p.assign(N + 1, 0);
    X.assign(N + 1, vector<ll>(d, 0));
    for (int j = 1; j <= N; j++) {
        p[j] = inf.readInt();
        for (int k = 0; k < d; k++) X[j][k] = inf.readInt();
    }

    // ---- internal baseline B: one stage, all acts in input order ----
    // setup charged twice (once in makespan==load, once in S).
    ll durAll = 0, setAll = 0;
    for (int j = 1; j <= N; j++) durAll += p[j];
    for (int j = 1; j < N; j++) setAll += setup2(j, j + 1);
    ll B = durAll + 2 * setAll;
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's schedule: exactly M stage lines ----
    vector<char> seen(N + 1, 0);
    ll makespan = 0, totalSetup = 0;
    int placed = 0;
    for (int i = 0; i < M; i++) {
        int c = ouf.readInt(0, N, "stageCount");
        ll dur = 0, setu = 0;
        int prev = -1;
        for (int t = 0; t < c; t++) {
            int a = ouf.readInt(1, N, "actIndex");
            if (seen[a]) quitf(_wa, "act %d appears more than once", a);
            seen[a] = 1; placed++;
            dur += p[a];
            if (prev != -1) setu += setup2(prev, a);
            prev = a;
        }
        ll load = dur + setu;
        if (load > makespan) makespan = load;
        totalSetup += setu;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (placed != N)
        quitf(_wa, "scheduled %d acts but N=%d (every act must appear exactly once)", placed, N);

    ll F = makespan + totalSetup;
    if (F <= 0) quitf(_wa, "non-positive objective F=%lld", F);

    double sc = min(1000.0, 100.0 * (double)B / (double)max((ll)1, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
