// TIER: trivial
// Scatter baseline == exactly what the checker measures as B: seed the k
// smallest-degree people (ties by smaller index). Scores ratio ~ 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, r, k;
    if (scanf("%d %d %d %d", &n, &m, &r, &k) != 4) return 0;
    vector<int> deg(n + 1, 0);
    for (int i = 0; i < m; i++) {
        int u, v; scanf("%d %d", &u, &v);
        deg[u]++; deg[v]++;
    }
    vector<int> order(n);
    for (int v = 1; v <= n; v++) order[v - 1] = v;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (deg[a] != deg[b]) return deg[a] < deg[b];
        return a < b;
    });
    int kk = min(k, n);
    printf("%d\n", kk);
    for (int i = 0; i < kk; i++) printf("%d\n", order[i]);
    return 0;
}
