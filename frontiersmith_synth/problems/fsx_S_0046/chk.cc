#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
struct Clause { ll w; vector<int> lits; };
vector<Clause> cls;

// total satisfied weight under assignment a[1..n] (a[i] in {0,1})
ll satWeight(const vector<int>& a) {
    ll tot = 0;
    for (auto& c : cls) {
        bool sat = false;
        for (int lit : c.lits) {
            int v = abs(lit);
            if (lit > 0) { if (a[v] == 1) { sat = true; break; } }
            else         { if (a[v] == 0) { sat = true; break; } }
        }
        if (sat) tot += c.w;
    }
    return tot;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    cls.resize(m);
    for (int i = 0; i < m; i++) {
        ll w = inf.readInt();
        int L = inf.readInt();
        cls[i].w = w;
        cls[i].lits.resize(L);
        for (int j = 0; j < L; j++) {
            int lit = inf.readInt();
            cls[i].lits[j] = lit;
        }
    }

    // internal baseline: the shipped all-0 calibration
    vector<int> zero(n + 1, 0);
    ll B = satWeight(zero);
    if (B <= 0) quitf(_fail, "bad instance: baseline B=%lld (no requirement met by all-0)", B);

    // ---- read & validate participant's assignment ----
    vector<int> a(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        a[i] = ouf.readInt(0, 1, format("mode[%d]", i).c_str());
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens after %d modes", n);

    ll F = satWeight(a);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
