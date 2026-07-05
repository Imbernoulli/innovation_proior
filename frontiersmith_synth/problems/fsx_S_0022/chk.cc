#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll>          W;      // clause weights
vector<vector<int>> LIT;    // clause literals (signed, |x| in 1..n)

// total satisfied weight under assignment val[1..n] (0/1)
ll satWeight(const vector<int>& val) {
    ll F = 0;
    for (int j = 0; j < m; j++) {
        bool sat = false;
        for (int l : LIT[j]) {
            int v = abs(l);
            bool want = (l > 0);            // +g -> want high-flow(1); -g -> want low-flow(0)
            if ((val[v] == 1) == want) { sat = true; break; }
        }
        if (sat) F += W[j];
    }
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    W.assign(m, 0);
    LIT.assign(m, {});
    for (int j = 0; j < m; j++) {
        ll w = inf.readInt();
        int k = inf.readInt();
        W[j] = w;
        LIT[j].reserve(k);
        for (int i = 0; i < k; i++) {
            int c = inf.readInt();
            LIT[j].push_back(c);
        }
    }

    // internal baseline: all-low-flow regime (every gate = 0)
    vector<int> allLow(n + 1, 0);
    ll B = satWeight(allLow);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld (must be >0)", B);

    // ---- read & validate participant's regime assignment ----
    vector<int> val(n + 1, 0);
    for (int g = 1; g <= n; g++)
        val[g] = ouf.readInt(0, 1, "gateRegime");
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d values)", n);

    ll F = satWeight(val);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
