// TIER: invalid
// Deliberately infeasible: prints one beacon whose last symbol is q (one past
// the valid alphabet [0,q-1]). The checker's bounded read rejects this
// immediately -> must score 0 regardless of the rest of the instance.
#include <cstdio>
int main() {
    int q, n, r, m, M, Kmax;
    if (scanf("%d %d %d %d %d %d", &q, &n, &r, &m, &M, &Kmax) != 6) return 0;
    printf("1\n");
    for (int i = 0; i < n; i++) {
        int v = (i + 1 < n) ? 0 : q; // out-of-range symbol on the last column
        printf("%d%c", v, i + 1 < n ? ' ' : '\n');
    }
    return 0;
}
