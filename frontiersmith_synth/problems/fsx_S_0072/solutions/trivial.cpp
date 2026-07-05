// TIER: trivial
// Baseline the checker measures: every ship works its earliest w_j hours
// contiguously (one run) entirely on shore power -> F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int T, J, G;
    if (scanf("%d %d %d", &T, &J, &G) != 3) return 0;
    vector<int> sp(T), od(T), C(T);
    for (int t = 0; t < T; t++) scanf("%d %d %d", &sp[t], &od[t], &C[t]);
    for (int j = 0; j < J; j++) {
        int a, b, w; scanf("%d %d %d", &a, &b, &w);
        for (int t = a; t < a + w; t++) printf("%d 1 ", t);
        printf("\n");
    }
    return 0;
}
