#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Single-Rack LIFO Picking Robot -- selective pickup/delivery tour generator.
// testId is a difficulty/structure ladder (1 tiny -> 10 large/adversarial).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // P grows with testId; medium scale.
    int Ptable[11]  = {0, 3, 6, 10, 16, 24, 34, 46, 58, 70, 80};
    int P = Ptable[min(max(testId,1),10)];

    int H = 2 + (testId % 6);          // rack height varies 2..7
    if (H < 1) H = 1; if (H > 12) H = 12;

    int C = 1000;                      // coordinate range [0,C]
    int x0 = C/2, y0 = C/2;            // dock near center

    // Structure regimes cycle through testIds to create diverse instances.
    int regime = testId % 5;
    // 0: uniform scatter, moderate penalties
    // 1: clustered shelves (batching pays off), penalties ~ solo cost
    // 2: penalties large (serve almost everyone)
    // 3: penalties small (serve few), long detours
    // 4: mixed hot spots

    // number of clusters for clustered regimes
    int K = max(2, (int)round(sqrt((double)P)));
    vector<pair<int,int>> centers(K);
    for (int k = 0; k < K; k++)
        centers[k] = {rnd.next(0, C), rnd.next(0, C)};

    auto clusterPt = [&](int spread)->pair<int,int>{
        int k = rnd.next(0, K-1);
        int cx = centers[k].first, cy = centers[k].second;
        int x = min(C, max(0, cx + rnd.next(-spread, spread)));
        int y = min(C, max(0, cy + rnd.next(-spread, spread)));
        return {x,y};
    };

    printf("%d %d\n", P, H);
    printf("%d %d\n", x0, y0);

    for (int i = 0; i < P; i++) {
        int px, py, dx, dy;
        if (regime == 1 || regime == 4) {
            auto s = clusterPt(regime == 1 ? 60 : 120);
            px = s.first; py = s.second;
            // delivery: a cluster point too (chutes grouped) for regime1, scattered for regime4
            if (regime == 1) { auto d = clusterPt(60); dx = d.first; dy = d.second; }
            else { dx = rnd.next(0, C); dy = rnd.next(0, C); }
        } else {
            px = rnd.next(0, C); py = rnd.next(0, C);
            dx = rnd.next(0, C); dy = rnd.next(0, C);
        }

        long long solo = (long long)abs(px - x0) + abs(py - y0)
                       + abs(dx - px) + abs(dy - py)
                       + abs(x0 - dx) + abs(y0 - dy);

        int w;
        if (regime == 0) {
            w = rnd.next(200, 4000);
        } else if (regime == 1) {
            // penalty near solo cost -> selection decision is delicate
            double f = rnd.next(60, 160) / 100.0;
            w = (int)llround(solo * f) + rnd.next(1, 200);
        } else if (regime == 2) {
            w = rnd.next(4000, 20000);       // serve almost everyone
        } else if (regime == 3) {
            w = rnd.next(1, 600);            // serve few
        } else {
            // regime 4 mixed: half cheap, half expensive
            if (rnd.next(0,1)) w = rnd.next(1, 800);
            else               w = rnd.next(3000, 20000);
        }
        if (w < 1) w = 1; if (w > 20000) w = 20000;

        printf("%d %d %d %d %d\n", px, py, dx, dy, w);
    }
    return 0;
}
