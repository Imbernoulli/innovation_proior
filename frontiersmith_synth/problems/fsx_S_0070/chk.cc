#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<int> cw;              // clause weight
vector<vector<int>> clit;   // clause literals (signed)

// total satisfied weight for assignment a[1..n] (a[i] in {0,1})
ll satWeight(const vector<int>& a) {
    ll F = 0;
    for (int c = 0; c < m; c++) {
        bool sat = false;
        for (int L : clit[c]) {
            int v = abs(L);
            bool truth = (L > 0) ? (a[v] == 1) : (a[v] == 0);
            if (truth) { sat = true; break; }
        }
        if (sat) F += cw[c];
    }
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    cw.resize(m);
    clit.resize(m);
    for (int c = 0; c < m; c++) {
        int k = inf.readInt();
        int w = inf.readInt();
        cw[c] = w;
        clit[c].resize(k);
        for (int t = 0; t < k; t++) {
            int L = inf.readInt();
            clit[c][t] = L;
        }
    }

    // ---- internal baseline: the all-Side-Stage (all-zero) placement ----
    vector<int> zero(n + 1, 0);
    ll B = satWeight(zero);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld", B);

    // ---- read & validate the participant's placement ----
    vector<int> a(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        int v = ouf.readInt(0, 1, "a_i");
        a[i] = v;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens (expected exactly %d values)", n);

    ll F = satWeight(a);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
