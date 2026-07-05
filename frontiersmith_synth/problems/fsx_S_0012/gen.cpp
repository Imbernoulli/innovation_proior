#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int N = 3 + (t - 1) * 6;          // 3 .. 57
    if (N < 3) N = 3;
    if (N > 60) N = 60;
    const int D = 10, OD = 100;

    vector<int> d(N), k(N), W(N);
    int prev = 0;
    for (int i = 0; i < N; i++) {
        int gap = rnd.next(2, 6);
        d[i] = prev + gap;
        k[i] = rnd.next(1, gap);      // 1 <= k_i <= gap  => sum_{j<=i} k_j <= d_i
        W[i] = D * k[i];
        prev = d[i];
    }
    int T = d[N - 1];

    vector<int> cap(T + 1), sc(T + 1);
    for (int s = 1; s <= T; s++) {
        bool storm = (rnd.next(0, 5) == 0);   // ~17% storms
        if (storm) {
            cap[s] = 0;
            sc[s] = OD;                       // spot pointless during a storm
        } else {
            cap[s] = rnd.next(3, 18);
            sc[s] = cap[s] * rnd.next(2, 6) + rnd.next(0, 2);  // per-unit ~2..6 < OD/D=10
            if (sc[s] < 1) sc[s] = 1;
            if (sc[s] > 200) sc[s] = 200;
        }
    }

    printf("%d %d %d %d\n", N, T, D, OD);
    for (int s = 1; s <= T; s++) printf("%d %d\n", cap[s], sc[s]);
    for (int i = 0; i < N; i++) printf("%d %d\n", W[i], d[i]);
    return 0;
}
