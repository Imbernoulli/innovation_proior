#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Ridgeline Lift Grid: tower-capped minimum-length connected ski-lift network.
// testId is a difficulty/structure ladder: tiny at 1 (example scale) -> large/adversarial
// by 10. Stations are distinct integer (x,y,h) triples; caps in {2,3,4} with cap>=2 so a
// chain baseline is always feasible. Input order is shuffled so the chain baseline is a
// genuinely long path (a fair "do-nothing" reference). Elevation h is drawn with terrain
// structure (ridges / basins) so the vertical-penalised cost really bites.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {4, 18, 45, 80, 140, 210, 300, 400, 520, 600};
    int N = Ns[idx - 1];
    const int C = 10000;      // horizontal extent
    const int H = 3000;       // altitude extent

    // structure knobs vary along the ladder
    bool clustered = (idx % 2 == 0);        // even tests are spatially clustered
    int terrain = idx % 3;                  // 0 uniform h, 1 ridge (h~x), 2 basin (radial)

    set<tuple<int,int,int>> used;
    vector<array<int,3>> pts;               // x, y, h

    auto altitude = [&](int x, int y) -> int {
        int h;
        if (terrain == 1) {
            // ridge: altitude rises with x, plus local noise
            h = (int)((double)x / C * H) + rnd.next(-250, 250);
        } else if (terrain == 2) {
            // basin: altitude grows with distance from the map centre
            double dx = x - C / 2.0, dy = y - C / 2.0;
            double r = sqrt(dx * dx + dy * dy) / (0.70711 * C);   // 0..~1
            h = (int)(r * H) + rnd.next(-250, 250);
        } else {
            h = rnd.next(0, H);
        }
        return min(H, max(0, h));
    };

    if (!clustered) {
        while ((int)pts.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            int h = altitude(x, y);
            if (used.insert({x, y, h}).second) pts.push_back({x, y, h});
        }
    } else {
        int K = max(2, N / 15);
        vector<pair<int,int>> ctr;
        for (int i = 0; i < K; i++)
            ctr.push_back({rnd.next(600, C - 600), rnd.next(600, C - 600)});
        int spread = 550;
        while ((int)pts.size() < N) {
            auto c = ctr[rnd.next(0, K - 1)];
            int x = min(C, max(0, c.first + rnd.next(-spread, spread)));
            int y = min(C, max(0, c.second + rnd.next(-spread, spread)));
            int h = altitude(x, y);
            if (used.insert({x, y, h}).second) pts.push_back({x, y, h});
        }
    }

    // shuffle so input order (used by the chain baseline) is arbitrary
    shuffle(pts.begin(), pts.end());

    // tower caps in {2,3,4}; later tests apply more cap-2 pressure so caps genuinely bind
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int r = rnd.next(0, 99);
        if (idx <= 3) {
            cap[i] = (r < 15) ? 2 : ((r < 85) ? 3 : 4);
        } else {
            cap[i] = (r < 35) ? 2 : ((r < 82) ? 3 : 4);
        }
    }

    printf("%d\n", N);
    for (int i = 0; i < N; i++)
        printf("%d %d %d %d\n", pts[i][0], pts[i][1], pts[i][2], cap[i]);
    return 0;
}
