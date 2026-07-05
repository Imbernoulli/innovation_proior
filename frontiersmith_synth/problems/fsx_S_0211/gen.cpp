#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Orbital Downlink Backbone: degree-capped Steiner relay network.
// testId is a difficulty/structure ladder: tiny uniform at 1 -> large clustered/skewed by
// 10. Assets are distinct integer points; caps in {2,3,4} with cap>=2 so the station-chain
// baseline is always feasible. The first K assets are ground stations (terminals), the rest
// are optional relay masts. Input order is shuffled so the chain baseline is a genuinely
// long path (a fair "do-nothing" reference), and masts are interspersed among the plain.
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {6, 16, 30, 55, 90, 140, 200, 280, 360, 450};
    int N = Ns[idx - 1];
    const int C = 10000;
    bool clustered = (idx % 2 == 0);   // even tests are clustered (worst-case for paths)

    // fraction of assets that are ground stations (terminals) varies across the ladder
    double fracs[10] = {0.67, 0.55, 0.6, 0.5, 0.65, 0.45, 0.55, 0.5, 0.6, 0.5};
    int K = (int)llround(N * fracs[idx - 1]);
    if (K < 3) K = 3;
    if (K > N) K = N;

    set<pair<int,int>> used;
    vector<pair<int,int>> pts;

    if (!clustered) {
        while ((int)pts.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    } else {
        int G = max(2, N / 15);
        vector<pair<int,int>> ctr;
        for (int i = 0; i < G; i++)
            ctr.push_back({rnd.next(600, C - 600), rnd.next(600, C - 600)});
        int spread = 500;
        while ((int)pts.size() < N) {
            auto c = ctr[rnd.next(0, G - 1)];
            int x = c.first + rnd.next(-spread, spread);
            int y = c.second + rnd.next(-spread, spread);
            x = min(C, max(0, x));
            y = min(C, max(0, y));
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    }

    // shuffle so which points become terminals (first K) and the chain order are arbitrary
    shuffle(pts.begin(), pts.end());

    // port caps in {2,3,4}; distribution tightens on later tests so caps genuinely bind
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int r = rnd.next(0, 99);
        if (idx <= 3) {
            cap[i] = (r < 15) ? 2 : ((r < 85) ? 3 : 4);
        } else {
            cap[i] = (r < 30) ? 2 : ((r < 80) ? 3 : 4);
        }
    }

    printf("%d %d\n", N, K);
    for (int i = 0; i < N; i++)
        printf("%d %d %d\n", pts[i].first, pts[i].second, cap[i]);
    return 0;
}
