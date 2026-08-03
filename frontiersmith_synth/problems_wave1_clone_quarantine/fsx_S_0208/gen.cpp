#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll cheb(ll ax, ll ay, ll bx, ll by) {
    return max(llabs(ax - bx), llabs(ay - by));
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int P = 8 + (t - 1) * 13;              // 8 .. 125
    int C = 200 + t * 140;                 // 340 .. 1600 coordinate span
    int qmax = 6;
    int Q = rnd.next(qmax, qmax * 4);      // capacity in [6,24], always >= max demand

    int x0 = rnd.next(0, C), y0 = rnd.next(0, C);

    // penalty regime varies by testId so different tests stress selection vs. ordering
    double blo, bhi;
    if (t % 3 == 0)      { blo = 0.7; bhi = 2.0; } // mixed
    else if (t % 3 == 1) { blo = 1.0; bhi = 1.8; } // mostly profitable -> ordering matters
    else                 { blo = 0.4; bhi = 1.3; } // marginal -> selection & batching matter

    // cluster centers create batching opportunities on the rail bridge
    int nc = 1 + t;
    vector<pair<int,int>> cen;
    for (int i = 0; i < nc; i++) cen.push_back({rnd.next(0, C), rnd.next(0, C)});

    auto pick = [&]() -> pair<int,int> {
        if (rnd.next(0, 2) == 0) {
            auto c = cen[rnd.next(0, (int)cen.size() - 1)];
            int r = C / 10 + 1;
            int x = min(C, max(0, c.first  + rnd.next(-r, r)));
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
        // instrumentation cost: small fixed settling cost, scaled to the map
        int c = rnd.next(0, C / 12 + 1);
        // solo round-trip cost bound (Chebyshev) used to size the omission penalty
        ll s = cheb(p.first, p.second, x0, y0)
             + cheb(p.first, p.second, d.first, d.second)
             + cheb(d.first, d.second, x0, y0)
             + c;
        double beta = rnd.next(blo, bhi);
        ll w = max(1LL, (ll)llround(beta * (double)s));
        printf("%d %d %d %d %d %d %lld\n", p.first, p.second, d.first, d.second, q, c, w);
    }
    return 0;
}
