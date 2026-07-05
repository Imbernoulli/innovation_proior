// TIER: greedy
// Value/cost greedy: sort parcels by value / cheapest-cost descending, assign
// each to its feasible minimum-cost drone.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    vector<long long> c(m + 1), rem(m + 1);
    for (int i = 1; i <= m; i++) { scanf("%lld", &c[i]); rem[i] = c[i]; }
    vector<vector<long long>> e(m + 1, vector<long long>(n + 1));
    vector<long long> w(n + 1);
    for (int j = 1; j <= n; j++) {
        scanf("%lld", &w[j]);
        for (int i = 1; i <= m; i++) scanf("%lld", &e[i][j]);
    }

    vector<int> order(n);
    for (int j = 0; j < n; j++) order[j] = j + 1;
    auto mincost = [&](int j) {
        long long mn = LLONG_MAX;
        for (int i = 1; i <= m; i++) mn = min(mn, e[i][j]);
        return mn;
    };
    sort(order.begin(), order.end(), [&](int x, int y) {
        double rx = (double)w[x] / (double)mincost(x);
        double ry = (double)w[y] / (double)mincost(y);
        if (rx != ry) return rx > ry;
        return w[x] > w[y];
    });

    vector<int> a(n + 1, 0);
    for (int j : order) {
        int best = 0; long long bestc = LLONG_MAX;
        for (int i = 1; i <= m; i++) {
            if (rem[i] >= e[i][j] && e[i][j] < bestc) { bestc = e[i][j]; best = i; }
        }
        if (best) { rem[best] -= e[best][j]; a[j] = best; }
    }
    for (int j = 1; j <= n; j++) printf("%d%c", a[j], j == n ? '\n' : ' ');
    return 0;
}
