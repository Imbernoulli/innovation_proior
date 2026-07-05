// TIER: invalid
// Deliberately infeasible: dispatches forager 1 to an out-of-range patch (P+1).
#include <bits/stdc++.h>
using namespace std;
int main() {
    int P, B;
    scanf("%d %d", &P, &B);
    long long s;
    for (int j = 1; j <= P; j++) scanf("%lld", &s);
    for (int i = 1; i <= B; i++) {
        int m; scanf("%d", &m);
        for (int k = 0; k < m; k++) { int p, a; scanf("%d %d", &p, &a); }
    }
    // forager 1 -> non-existent patch P+1 -> feasibility violation -> score 0
    printf("%d", P + 1);
    for (int i = 2; i <= B; i++) printf(" 0");
    printf("\n");
    return 0;
}
