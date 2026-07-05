// TIER: invalid
// Deliberately infeasible: prints an out-of-range placement value (2) for act 1,
// which the checker rejects (each value must be 0 or 1) -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int c = 0; c < m; c++) {
        int k, w; scanf("%d %d", &k, &w);
        for (int t = 0; t < k; t++) { int L; scanf("%d", &L); }
    }
    // first value out of range (2 instead of 0/1)
    printf("2");
    for (int i = 1; i < n; i++) printf(" 0");
    printf("\n");
    return 0;
}
