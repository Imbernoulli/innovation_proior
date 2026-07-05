// TIER: greedy
// Independent per-gate weighted vote: for each gate, set it to high-flow if the
// total run weight of positive (+g) conditions on it exceeds that of negative
// (-g) conditions, else low-flow. One pass, ignores clause interactions.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<ll> wPos(n + 1, 0), wNeg(n + 1, 0);
    for (int j = 0; j < m; j++) {
        int w, k; scanf("%d %d", &w, &k);
        for (int i = 0; i < k; i++) {
            int c; scanf("%d", &c);
            if (c > 0) wPos[c] += w; else wNeg[-c] += w;
        }
    }
    for (int g = 1; g <= n; g++) {
        int v = (wPos[g] > wNeg[g]) ? 1 : 0;
        printf(g == 1 ? "%d" : " %d", v);
    }
    printf("\n");
    return 0;
}
