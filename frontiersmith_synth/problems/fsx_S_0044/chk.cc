#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m;
vector<array<int, 3>> edges; // u, v, w

ll cutOf(const vector<int>& side) {
    ll F = 0;
    for (auto& e : edges)
        if (side[e[0]] != side[e[1]]) F += e[2];
    return F;
}

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    n = inf.readInt();
    m = inf.readInt();
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u = inf.readInt();
        int v = inf.readInt();
        int w = inf.readInt();
        edges[i] = {u, v, w};
    }

    // ---- internal baseline: the alternating dock  a_i = i % 2  (odd->1, even->0) ----
    vector<int> base(n + 1, 0);
    for (int i = 1; i <= n; i++) base[i] = i % 2;
    ll B = cutOf(base);
    if (B <= 0) quitf(_fail, "bad instance: baseline cut B=%lld", B);

    // ---- read & validate participant's balanced assignment ----
    vector<int> side(n + 1, 0);
    int ones = 0;
    for (int i = 1; i <= n; i++) {
        int a = ouf.readInt(0, 1, "a_i");
        side[i] = a;
        ones += a;
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");
    if (ones != n / 2)
        quitf(_wa, "assignment not balanced: %d asteroids at refinery 1, need %d", ones, n / 2);

    ll F = cutOf(side);

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
