#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    int idx = t; if (idx < 1) idx = 1; if (idx > 10) idx = 10;
    int Ns[10] = {6, 12, 25, 50, 100, 200, 400, 800, 1400, 2000};
    int N = Ns[idx - 1];
    ll C = 1000000;

    printf("%d\n", N);
    // depot at meadow centre
    printf("%lld %lld\n", C / 2, C / 2);

    // structural variety: some instances uniform, some clustered, some skewed penalties
    int mode = idx % 3;               // 0 clustered, 1 uniform, 2 mixed
    int K = 4 + (idx % 5);            // number of clusters when clustered
    vector<ll> cx(K), cy(K);
    for (int k = 0; k < K; k++) { cx[k] = rnd.next(0LL, C); cy[k] = rnd.next(0LL, C); }

    // comb cell sits within a local radius of its flower patch, so the visiting
    // ORDER of tasks (the connecting legs) is the dominant, reducible cost.
    ll dr = C / 8;
    for (int i = 0; i < N; i++) {
        ll px, py, dx, dy, w;
        if (mode == 0) {
            int a = rnd.next(0, K - 1);
            ll r = C / 12;
            px = min(C, max(0LL, cx[a] + rnd.next(-r, r)));
            py = min(C, max(0LL, cy[a] + rnd.next(-r, r)));
        } else {
            px = rnd.next(0LL, C); py = rnd.next(0LL, C);
        }
        dx = min(C, max(0LL, px + rnd.next(-dr, dr)));
        dy = min(C, max(0LL, py + rnd.next(-dr, dr)));
        // penalties: mixed => some cheap-to-skip, some expensive; uniform => moderate
        if (mode == 2) {
            if (rnd.next(0, 2) == 0) w = rnd.next(0LL, C);        // cheap: skipping tempting
            else                    w = rnd.next(2 * C, 5 * C);   // expensive: must serve
        } else {
            w = rnd.next(C / 2, 4 * C);
        }
        printf("%lld %lld %lld %lld %lld\n", px, py, dx, dy, w);
    }
    return 0;
}
