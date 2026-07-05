#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    long long Ts[11] = {0, 6, 200, 800, 2000, 5000, 12000, 30000, 60000, 120000, 200000};
    long long T = (t >= 1 && t <= 10) ? Ts[t] : 1000;
    // fixed lift economics (invariants: S<M, P<D, a<=M/2)
    long long M = 200, S = 80, D = 100, P = 30, h = 1, H = 1000000;
    long long AMAX = 40; // <= M/2
    // vary spot availability density across the ladder to create different gap structures
    int pav_permille = 300 + 40 * ((t * 7) % 8); // 300..580
    printf("%lld %lld %lld %lld %lld %lld %lld\n", T, M, S, D, P, h, H);
    for (long long i = 0; i < T; i++) {
        long long a = rnd.next(0LL, AMAX);
        int av = (rnd.next(0, 999) < pav_permille) ? 1 : 0;
        printf("%lld %d\n", a, av);
    }
    return 0;
}
