// TIER: invalid
// Deliberately infeasible: bends station 1 for every line instead of a
// permutation of 1..m -- the checker's "bent more than once" guard rejects
// it -> no Ratio printed by a normal run path -> scores 0.
#include <cstdio>
int main() {
    int m; long long L; int c, TOL; long long K;
    scanf("%d %lld %d %d %lld", &m, &L, &c, &TOL, &K);
    for (int i = 1; i <= m; i++) {
        long long x, theta, delta, w;
        scanf("%lld %lld %lld %lld", &x, &theta, &delta, &w);
    }
    for (int i = 1; i <= m; i++) printf("1 0\n");
    return 0;
}
