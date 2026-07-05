// TIER: invalid
// Deliberately infeasible: emits an out-of-range mode value (2) -> checker rejects -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    for (int i = 0; i < m; i++) {
        int w, L; scanf("%d %d", &w, &L);
        for (int j = 0; j < L; j++) { int lit; scanf("%d", &lit); }
    }
    printf("2"); // invalid mode: outside {0,1}
    for (int i = 1; i < n; i++) printf(" 0");
    printf("\n");
    return 0;
}
