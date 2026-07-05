// TIER: invalid
// Deliberately infeasible: start every leg at time 0. Any mission with >= 2 legs
// violates precedence (leg 1 starts at 0 < end of leg 0), so the checker rejects
// it and the score is 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> o(n);
    for (int j = 0; j < n; j++) {
        scanf("%d", &o[j]);
        for (int k = 0; k < o[j]; k++) { int a, b; scanf("%d %d", &a, &b); }
    }
    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("0%c", k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
