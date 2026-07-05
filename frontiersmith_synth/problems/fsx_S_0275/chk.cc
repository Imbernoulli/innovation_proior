#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();

    vector<ll> w(n + 1);
    for (int i = 1; i <= n; i++) w[i] = inf.readInt();

    // store conflict edges in a hash set for O(1) lookup
    unordered_set<ll> eset;
    eset.reserve(2 * (size_t)m + 8);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt(1, n, "u");
        int v = inf.readInt(1, n, "v");
        if (u > v) swap(u, v);
        if (u != v) eset.insert((ll)u * (n + 1) + v);
    }

    // internal baseline: the single most valuable site (always feasible on its own)
    ll B = 0;
    for (int i = 1; i <= n; i++) B = max(B, w[i]);
    if (B <= 0) quitf(_fail, "bad instance: B=%lld", B);

    // ---- read & validate participant's chosen set ----
    int c = ouf.readInt(0, n, "c");
    vector<int> sel(c);
    vector<char> chosen(n + 1, 0);
    ll F = 0;
    for (int i = 0; i < c; i++) {
        int v = ouf.readInt(1, n, "site");
        if (chosen[v]) quitf(_wa, "site %d selected more than once", v);
        chosen[v] = 1;
        sel[i] = v;
        F += w[v];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    // validate independence: no chosen pair may be a conflict edge
    for (int i = 0; i < c; i++) {
        for (int j = i + 1; j < c; j++) {
            int a = sel[i], b = sel[j];
            if (a > b) swap(a, b);
            if (eset.count((ll)a * (n + 1) + b))
                quitf(_wa, "chosen sites %d and %d conflict", sel[i], sel[j]);
        }
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
