// TIER: invalid
// Deliberately infeasible: emits observation 1 n times (duplicates, not a
// permutation). The checker must reject this and score 0.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, L;
    if (scanf("%d %d", &n, &L) != 2) return 0;
    for (int i = 1; i <= n; i++) {
        int s; scanf("%d", &s);
        for (int j = 0; j < s; j++) { int p; scanf("%d", &p); }
    }
    for (int i = 0; i < n; i++) printf("1%c", i == n - 1 ? '\n' : ' ');
    return 0;
}
