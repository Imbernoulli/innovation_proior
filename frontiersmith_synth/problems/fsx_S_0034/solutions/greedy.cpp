// TIER: greedy
// Weighted majority vote: for each ride, sum the weight of requests that want it
// forward vs. reverse; set it to the heavier side. One pass, ignores interactions.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<ll> posW(n + 1, 0), negW(n + 1, 0);
    for (int c = 0; c < m; c++) {
        ll w; int k;
        scanf("%lld %d", &w, &k);
        for (int j = 0; j < k; j++) {
            int lit; scanf("%d", &lit);
            int v = abs(lit);
            if (lit > 0) posW[v] += w; else negW[v] += w;
        }
    }
    for (int v = 1; v <= n; v++) {
        int x = (posW[v] > negW[v]) ? 1 : 0;
        printf("%s%d", v > 1 ? " " : "", x);
    }
    printf("\n");
    return 0;
}
