// TIER: trivial
// Send only the single best-value tour (group->gallery with max profit that fits
// alone). F == B by construction -> ratio == 0.1, the calibration baseline.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int V, G;
    if (scanf("%d %d", &V, &G) != 2) return 0;
    vector<int> C(G + 1);
    for (int j = 1; j <= G; j++) scanf("%d", &C[j]);
    int bi = -1, bj = -1, bp = -1;
    for (int i = 1; i <= V; i++)
        for (int j = 1; j <= G; j++) {
            int a, p; scanf("%d %d", &a, &p);
            if (a <= C[j] && p > bp) { bp = p; bi = i; bj = j; }
        }
    if (bi < 0) { printf("0\n"); return 0; }
    printf("1\n%d %d\n", bi, bj);
    return 0;
}
