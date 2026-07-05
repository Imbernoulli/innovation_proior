#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef unsigned long long u64;

int n, m;
vector<int> eu, ev;
vector<ll> ew;

// cut weight of an assignment x[1..n]
ll cutOf(const vector<int>& x) {
    ll F = 0;
    for (int e = 0; e < m; e++)
        if (x[eu[e]] != x[ev[e]]) F += ew[e];
    return F;
}

// deterministic reference partition R (must match the statement + trivial.cpp)
vector<int> referencePartition() {
    vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i + 1;         // cluster ids 1..n
    const u64 MUL = 11400714819323198485ULL;            // 0x9E3779B97F4A7C15
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        u64 ka = (u64)a * MUL, kb = (u64)b * MUL;
        if (ka != kb) return ka < kb;
        return a < b;
    });
    vector<int> x(n + 1, 0);
    int half = n / 2;                                    // first floor(n/2) -> crew 0
    for (int r = 0; r < n; r++)
        x[idx[r]] = (r < half) ? 0 : 1;
    return x;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    eu.resize(m); ev.resize(m); ew.resize(m);
    for (int e = 0; e < m; e++) {
        eu[e] = inf.readInt();
        ev[e] = inf.readInt();
        ew[e] = inf.readInt();
    }

    // ---- internal baseline: the reference scrambled balanced partition ----
    vector<int> ref = referencePartition();
    ll B = cutOf(ref);
    if (B <= 0) quitf(_fail, "bad instance: reference cut B=%lld is not positive", B);

    // ---- read & validate participant's assignment ----
    vector<int> x(n + 1, 0);
    ll c0 = 0, c1 = 0;
    for (int i = 1; i <= n; i++) {
        int xi = ouf.readInt(0, 1, "crew");
        x[i] = xi;
        if (xi == 0) c0++; else c1++;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    if (llabs(c0 - c1) > 1)
        quitf(_wa, "unbalanced crews: c0=%lld c1=%lld differ by more than 1", c0, c1);

    ll F = cutOf(x);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
