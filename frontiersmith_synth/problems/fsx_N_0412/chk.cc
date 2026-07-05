#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<ll> w;
vector<int> rr;
vector<vector<pair<int,int>>> cl; // (var, sign): sign=+1 positive literal, -1 negative

// count nectar (sum of w_j over routes with met-count ≡ r_j mod 3) for assignment x[1..n]
ll countWeight(const vector<int>& x) {
    ll tot = 0;
    for (int j = 0; j < m; j++) {
        int c = 0;
        for (auto& p : cl[j]) {
            int val = x[p.first];
            bool sat = (p.second > 0) ? (val == 1) : (val == 0);
            if (sat) c++;
        }
        if (c % 3 == rr[j]) tot += w[j];
    }
    return tot;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    w.resize(m);
    rr.resize(m);
    cl.assign(m, {});
    for (int j = 0; j < m; j++) {
        w[j]  = inf.readLong();
        rr[j] = inf.readInt();
        int k = inf.readInt();
        cl[j].reserve(k);
        for (int t = 0; t < k; t++) {
            int lit = inf.readInt();
            int v = abs(lit);
            int s = (lit > 0) ? 1 : -1;
            cl[j].push_back({v, s});
        }
    }

    // internal baseline: all-WEST assignment (every bee 0)
    vector<int> zero(n + 1, 0);
    ll B = countWeight(zero);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld is not positive", B);

    // ---- read & validate participant's headings ----
    vector<int> x(n + 1, 0);
    for (int v = 1; v <= n; v++) {
        x[v] = ouf.readInt(0, 1, "x");
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after n headings");

    ll F = countWeight(x);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
