#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Prospector Relay Grid: degree-capped minimum-length connected network.
// testId is a difficulty/structure ladder: tiny uniform at 1 -> large clustered/skewed
// caps by 10. Points are distinct integer coordinates; caps in {2,3,4} with cap>=2 so a
// chain baseline is always feasible. Input order is shuffled so the chain baseline is a
// genuinely long path (a fair "do-nothing" reference).
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Ns[10] = {8, 20, 40, 70, 110, 160, 230, 320, 450, 600};
    int N = Ns[idx - 1];
    const int C = 10000;
    bool clustered = (idx % 2 == 0);   // even tests are clustered (worst-case for paths)

    set<pair<int,int>> used;
    vector<pair<int,int>> pts;

    if (!clustered) {
        while ((int)pts.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    } else {
        int K = max(2, N / 15);
        vector<pair<int,int>> ctr;
        for (int i = 0; i < K; i++)
            ctr.push_back({rnd.next(600, C - 600), rnd.next(600, C - 600)});
        int spread = 500;
        while ((int)pts.size() < N) {
            auto c = ctr[rnd.next(0, K - 1)];
            int x = c.first + rnd.next(-spread, spread);
            int y = c.second + rnd.next(-spread, spread);
            x = min(C, max(0, x));
            y = min(C, max(0, y));
            if (used.insert({x, y}).second) pts.push_back({x, y});
        }
    }

    // shuffle so input order (used by the chain baseline) is arbitrary
    shuffle(pts.begin(), pts.end());

    // relay-head caps in {2,3,4}; distribution varies with testId
    vector<int> cap(N);
    for (int i = 0; i < N; i++) {
        int r = rnd.next(0, 99);
        if (idx <= 3) {
            // early tests: mostly tight cap 3
            cap[i] = (r < 15) ? 2 : ((r < 85) ? 3 : 4);
        } else {
            // later tests: more cap-2 pressure -> caps genuinely bind
            cap[i] = (r < 30) ? 2 : ((r < 80) ? 3 : 4);
        }
    }

    printf("%d\n", N);
    for (int i = 0; i < N; i++)
        printf("%d %d %d\n", pts[i].first, pts[i].second, cap[i]);
    return 0;
}
