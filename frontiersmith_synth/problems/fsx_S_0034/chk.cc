#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> wt;                 // clause weight
vector<vector<int>> cl;       // clause -> signed literals

// weight of clauses satisfied by an assignment x (x[v] in {0,1}, v 1..n)
ll satWeight(const vector<int>& x) {
    ll F = 0;
    for (int c = 0; c < m; c++) {
        bool sat = false;
        for (int lit : cl[c]) {
            int v = abs(lit);
            bool litTrue = (lit > 0) ? (x[v] == 1) : (x[v] == 0);
            if (litTrue) { sat = true; break; }
        }
        if (sat) F += wt[c];
    }
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    wt.assign(m, 0);
    cl.assign(m, {});
    for (int c = 0; c < m; c++) {
        ll w = inf.readLong();
        int k = inf.readInt();
        wt[c] = w;
        cl[c].reserve(k);
        for (int j = 0; j < k; j++) {
            int lit = inf.readInt();
            cl[c].push_back(lit);
        }
    }

    // internal baseline: all-reverse (all-zero) assignment
    vector<int> zero(n + 1, 0);
    ll B = satWeight(zero);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate participant assignment ----
    vector<int> x(n + 1, 0);
    for (int v = 1; v <= n; v++) {
        x[v] = ouf.readInt(0, 1, format("x[%d]", v).c_str());
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d values)", n);

    ll F = satWeight(x);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
