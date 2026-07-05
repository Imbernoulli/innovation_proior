// TIER: greedy
// Per-switch weighted majority: for each switch, sum requirement weight where it
// appears as a positive literal vs a negative literal; set it to the heavier side.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<long long> posW(n + 1, 0), negW(n + 1, 0);
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        for (int i = 0; i < k; i++) {
            int L; scanf("%d", &L);
            int v = abs(L);
            if (L > 0) posW[v] += w; else negW[v] += w;
        }
    }
    for (int v = 1; v <= n; v++) {
        int b = (posW[v] >= negW[v]) ? 1 : 0;
        printf("%d%c", b, v == n ? '\n' : ' ');
    }
    if (n == 0) printf("\n");
    return 0;
}
