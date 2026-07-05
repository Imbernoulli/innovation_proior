// TIER: greedy
// Value-density greedy: consider all (act,platform) pairs sorted by thrill-per-kilowatt
// descending; assign each act to the first still-affordable platform it appears with.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    vector<long long> C(m + 1);
    for (int j = 1; j <= m; j++) scanf("%lld", &C[j]);
    vector<vector<long long>> v(n + 1, vector<long long>(m + 1));
    vector<vector<long long>> w(n + 1, vector<long long>(m + 1));
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%lld", &v[i][j]);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) scanf("%lld", &w[i][j]);

    struct P { double d; int i, j; };
    vector<P> pairs;
    pairs.reserve((size_t)n * m);
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
            pairs.push_back({(double)v[i][j] / (double)w[i][j], i, j});
    sort(pairs.begin(), pairs.end(), [](const P& x, const P& y) {
        if (x.d != y.d) return x.d > y.d;
        if (x.i != y.i) return x.i < y.i;
        return x.j < y.j;
    });

    vector<long long> rem = C;
    vector<int> a(n + 1, 0);
    for (const auto& p : pairs) {
        if (a[p.i] != 0) continue;
        if (w[p.i][p.j] <= rem[p.j]) { rem[p.j] -= w[p.i][p.j]; a[p.i] = p.j; }
    }
    for (int i = 1; i <= n; i++) printf("%d%c", a[i], i == n ? '\n' : ' ');
    return 0;
}
