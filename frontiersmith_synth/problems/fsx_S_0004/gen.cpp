#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int P = 5 + (t - 1) * 6;          // 5 .. 59
    int C = 100 + t * 100;            // 200 .. 1200 coordinate span
    int qmax = 5;
    int Q = rnd.next(qmax, qmax * 3); // capacity in [5,15], always >= max quantity

    int x0 = rnd.next(0, C), y0 = rnd.next(0, C);

    // penalty regime varies by testId so different tests stress selection vs. ordering
    double blo, bhi;
    if (t % 3 == 0)      { blo = 0.6; bhi = 1.8; } // mixed
    else if (t % 3 == 1) { blo = 0.9; bhi = 1.6; } // mostly profitable -> ordering matters
    else                 { blo = 0.3; bhi = 1.1; } // marginal -> selection & batching matter

    // cluster centers to create batching opportunities
    int nc = 1 + t;
    vector<pair<int,int>> cen;
    for (int i = 0; i < nc; i++) cen.push_back({rnd.next(0, C), rnd.next(0, C)});

    auto pick = [&]() -> pair<int,int> {
        if (rnd.next(0, 2) == 0) {
            auto c = cen[rnd.next(0, (int)cen.size() - 1)];
            int r = C / 10 + 1;
            int x = min(C, max(0, c.first + rnd.next(-r, r)));
            int y = min(C, max(0, c.second + rnd.next(-r, r)));
            return {x, y};
        } else {
            return {rnd.next(0, C), rnd.next(0, C)};
        }
    };

    printf("%d %d\n", P, Q);
    printf("%d %d\n", x0, y0);
    for (int i = 0; i < P; i++) {
        auto p = pick();
        auto d = pick();
        int q = rnd.next(1, qmax);
        long long s = llabs((long long)p.first - x0) + llabs((long long)p.second - y0)
                    + llabs((long long)p.first - d.first) + llabs((long long)p.second - d.second)
                    + llabs((long long)d.first - x0) + llabs((long long)d.second - y0);
        double beta = rnd.next(blo, bhi);
        long long w = max(1LL, (long long)llround(beta * (double)s));
        printf("%d %d %d %d %d %lld\n", p.first, p.second, d.first, d.second, q, w);
    }
    return 0;
}
