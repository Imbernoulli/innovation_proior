#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Buried Cable Trenches: degree-capped Steiner-flavoured connectivity design.
// testId is a difficulty/structure ladder: tiny grid at 1 -> large mixed by 10.
// Markers 1..T are artifact chambers (must connect); T+1..N are optional relay posts.
// Distributions: 0 = dig lattice (grid-aligned, thematic), 1 = uniform, 2 = clustered.
// Cap regimes: 0 = all cap 2 (path-forced, hardest), 1 = {2,3}, 2 = {2,3,4}.
// Input order is shuffled within each group so the chamber-chain baseline is a genuine
// long path (a fair "do-nothing" reference).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {6, 20, 45, 90, 150, 240, 340, 450, 540, 600};
    int N = Ns[idx - 1];
    const int C = 10000;

    // fraction of markers that are artifact chambers: 0.55 .. 0.85
    double frac = 0.55 + 0.03 * ((idx * 7) % 11);
    if (frac > 0.9) frac = 0.9;
    int T = max(2, (int)llround(N * frac));
    if (T > N) T = N;

    int mode = idx % 3;              // 0 lattice, 1 uniform, 2 clustered
    int capmode = (idx + 1) % 3;     // decorrelated from mode

    set<pair<int,int>> used;
    vector<pair<int,int>> pts;

    auto tryadd = [&](int x, int y) {
        x = min(C, max(0, x));
        y = min(C, max(0, y));
        if (used.insert({x, y}).second) { pts.push_back({x, y}); return true; }
        return false;
    };

    if (mode == 0) {
        // dig lattice: place markers on distinct nodes of a coarse grid
        int g = 200;                 // 51 positions per axis -> 2601 nodes >= 600
        int span = C / g;            // 50
        while ((int)pts.size() < N) {
            int gx = rnd.next(0, span), gy = rnd.next(0, span);
            tryadd(gx * g, gy * g);
        }
    } else if (mode == 1) {
        while ((int)pts.size() < N) tryadd(rnd.next(0, C), rnd.next(0, C));
    } else {
        int K = max(2, N / 18);
        vector<pair<int,int>> ctr;
        for (int i = 0; i < K; i++)
            ctr.push_back({rnd.next(700, C - 700), rnd.next(700, C - 700)});
        int spread = 550;
        while ((int)pts.size() < N) {
            auto c = ctr[rnd.next(0, K - 1)];
            tryadd(c.first + rnd.next(-spread, spread),
                   c.second + rnd.next(-spread, spread));
        }
    }

    // shuffle chambers and posts separately so groups stay contiguous but order is arbitrary
    shuffle(pts.begin(), pts.begin() + T);
    if (T < N) shuffle(pts.begin() + T, pts.end());

    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int r = rnd.next(0, 99);
        if (capmode == 0) {
            cap[i] = 2;
        } else if (capmode == 1) {
            cap[i] = (r < 55) ? 2 : 3;
        } else {
            cap[i] = (r < 40) ? 2 : ((r < 80) ? 3 : 4);
        }
    }

    printf("%d %d\n", N, T);
    for (int i = 0; i < N; i++)
        printf("%d %d %d\n", pts[i].first, pts[i].second, cap[i]);
    return 0;
}
