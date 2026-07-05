// TIER: invalid
// Deliberately infeasible: routes EVERY beacon through station 1, blowing past its
// bandwidth budget -> checker rejects (station 1 over budget) -> score 0.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    vector<int> cap(m + 1);
    for (int i = 1; i <= m; i++) scanf("%d", &cap[i]);
    for (int j = 1; j <= n; j++)
        for (int i = 1; i <= m; i++) { int a, b; scanf("%d %d", &a, &b); }
    for (int j = 1; j <= n; j++) printf("1%c", j == n ? '\n' : ' ');
    return 0;
}
