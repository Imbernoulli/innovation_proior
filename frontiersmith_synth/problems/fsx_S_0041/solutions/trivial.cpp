// TIER: trivial
// Sequential fallback: every task on its first-listed asset, back-to-back on one
// global timeline.  Makespan == B, so this scores the calibration baseline ~0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    long long clockT = 0;
    for (int j = 0; j < J; j++) {
        int k; scanf("%d", &k);
        for (int t = 0; t < k; t++) {
            int e; scanf("%d", &e);
            int firstA = -1; long long firstD = 0;
            for (int r = 0; r < e; r++) {
                int a, d; scanf("%d %d", &a, &d);
                if (r == 0) { firstA = a; firstD = d; }
            }
            printf("%d %lld\n", firstA, clockT);
            clockT += firstD;
        }
    }
    return 0;
}
