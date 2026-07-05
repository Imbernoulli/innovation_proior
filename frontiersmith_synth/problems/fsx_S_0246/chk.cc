#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerTestlibCmd(argc, argv);

    int V = inf.readInt();
    int G = inf.readInt();
    vector<ll> C(G + 1);
    for (int j = 1; j <= G; j++) C[j] = inf.readInt();

    vector<vector<ll>> a(V + 1, vector<ll>(G + 1));
    vector<vector<ll>> p(V + 1, vector<ll>(G + 1));
    ll B = 0; // baseline: max profit of a single feasible tour
    for (int i = 1; i <= V; i++) {
        for (int j = 1; j <= G; j++) {
            a[i][j] = inf.readInt();
            p[i][j] = inf.readInt();
            if (a[i][j] <= C[j] && p[i][j] > B) B = p[i][j];
        }
    }
    if (B <= 0) quitf(_fail, "bad instance: no feasible single tour, B=%lld", B);

    // ---- read & validate participant's assignment ----
    int r = ouf.readInt(0, V, "r");
    vector<char> usedGroup(V + 1, 0);
    vector<ll> load(G + 1, 0);
    ll F = 0;
    for (int e = 0; e < r; e++) {
        int i = ouf.readInt(1, V, "groupIndex");
        int j = ouf.readInt(1, G, "galleryIndex");
        if (usedGroup[i]) quitf(_wa, "group %d assigned more than once", i);
        usedGroup[i] = 1;
        load[j] += a[i][j];
        if (load[j] > C[j])
            quitf(_wa, "gallery %d over budget: load %lld > C=%lld", j, load[j], C[j]);
        F += p[i][j];
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output tokens");

    double sc = min(1000.0, 100.0 * (double)F / (double)max((ll)1, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
