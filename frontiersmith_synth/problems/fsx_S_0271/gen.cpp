#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Lunar Habitat Pressure Loops: survivable (2-edge-connected) degree-capped
// minimum-length tunnel network. testId is a difficulty/structure ladder: tiny uniform at
// 1 -> large clustered / grid-perturbed / skewed caps by 10. Points have distinct integer
// coordinates; caps in {2,3,4} with cap>=2 so a Hamiltonian-cycle loop baseline is always
// feasible. Input order is shuffled so the loop baseline is a genuinely long cycle (a fair
// "do-nothing" reference for the score).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {6, 18, 40, 70, 110, 160, 220, 300, 400, 500};
    int N = Ns[idx - 1];
    const int C = 10000;
    int mode = idx % 3;  // 0 uniform, 1 clustered, 2 grid-perturbed

    set<pair<int,int>> used;
    vector<pair<int,int>> pts;

    if (mode == 0) {
        while ((int)pts.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    } else if (mode == 1) {
        int K = max(2, N / 15);
        vector<pair<int,int>> ctr;
        for (int k = 0; k < K; k++) ctr.push_back({rnd.next(500, C - 500), rnd.next(500, C - 500)});
        int spread = 350;
        while ((int)pts.size() < N) {
            auto c = ctr[rnd.next(0, K - 1)];
            int x = c.first + rnd.next(-spread, spread);
            int y = c.second + rnd.next(-spread, spread);
            x = min(C, max(0, x)); y = min(C, max(0, y));
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    } else {
        // perturbed lattice
        int side = (int)ceil(sqrt((double)N));
        int step = max(1, C / (side + 1));
        int jitter = max(1, step / 3);
        for (int gy = 0; gy < side && (int)pts.size() < N; gy++) {
            for (int gx = 0; gx < side && (int)pts.size() < N; gx++) {
                int x = (gx + 1) * step + rnd.next(-jitter, jitter);
                int y = (gy + 1) * step + rnd.next(-jitter, jitter);
                x = min(C, max(0, x)); y = min(C, max(0, y));
                if (used.insert({x, y}).second) pts.push_back({x, y});
            }
        }
        while ((int)pts.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    }

    // shuffle input order so the loop baseline (input-order cycle) is long
    shuffle(pts.begin(), pts.end());

    // caps in {2,3,4}: mostly 2 (tight ports) with a scatter of 3/4; a few tests are
    // uniformly capped at 2 (pure survivable-tour) to vary structure.
    bool tightAll2 = (idx % 4 == 3);

    printf("%d\n", N);
    for (int i = 0; i < N; i++) {
        int cap;
        if (tightAll2) cap = 2;
        else {
            int r = rnd.next(0, 99);
            cap = (r < 60) ? 2 : (r < 88) ? 3 : 4;
        }
        printf("%d %d %d\n", pts[i].first, pts[i].second, cap);
    }
    return 0;
}
