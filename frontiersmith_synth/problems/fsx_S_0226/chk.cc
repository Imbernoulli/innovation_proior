#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    int n = inf.readInt();
    int m = inf.readInt();
    int D = inf.readInt();

    vector<vector<pair<int,int>>> cue(m);   // (marker, expected color)
    vector<long long> pw(m);
    vector<long long> colorWeight(D, 0);     // weight corroborated by all-color-c

    for (int j = 0; j < m; j++) {
        int k = inf.readInt();
        cue[j].resize(k);
        // distinct colors present in this observation (dedup so uniform-c counts once)
        set<int> presentColors;
        for (int e = 0; e < k; e++) {
            int v = inf.readInt(1, n, "marker");
            int c = inf.readInt(0, D - 1, "color");
            cue[j][e] = {v, c};
            presentColors.insert(c);
        }
        long long w = inf.readInt();
        pw[j] = w;
        for (int c : presentColors) colorWeight[c] += w;
    }

    long long B = 0;
    for (int c = 0; c < D; c++) B = max(B, colorWeight[c]);

    // participant coloring: exactly n colors in [0, D-1], nothing else.
    vector<int> x(n + 1, 0);
    for (int i = 1; i <= n; i++)
        x[i] = ouf.readInt(0, D - 1, "coloring");
    if (!ouf.seekEof()) quitf(_wa, "trailing output after %d colors", n);

    long long F = 0;
    for (int j = 0; j < m; j++) {
        bool ok = false;
        for (auto& pr : cue[j])
            if (x[pr.first] == pr.second) { ok = true; break; }
        if (ok) F += pw[j];
    }

    double sc = min(1000.0, 100.0 * (double)F / (double)max(1LL, B));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
