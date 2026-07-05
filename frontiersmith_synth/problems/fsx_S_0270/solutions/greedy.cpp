// TIER: greedy
// Value-density greedy: rank every (beacon,station) option by value/bandwidth,
// then assign each beacon to the best-ranked station that still fits its budget.
#include <bits/stdc++.h>
using namespace std;
int main() {
    int m, n;
    if (scanf("%d %d", &m, &n) != 2) return 0;
    vector<int> cap(m + 1);
    for (int i = 1; i <= m; i++) scanf("%d", &cap[i]);
    vector<vector<int>> v(m + 1, vector<int>(n + 1)), w(m + 1, vector<int>(n + 1));
    for (int j = 1; j <= n; j++)
        for (int i = 1; i <= m; i++) scanf("%d %d", &v[i][j], &w[i][j]);

    // options sorted by descending value density
    struct Opt { double d; int val; int j; int i; };
    vector<Opt> opts;
    opts.reserve((size_t)m * n);
    for (int j = 1; j <= n; j++)
        for (int i = 1; i <= m; i++)
            opts.push_back({(double)v[i][j] / (double)w[i][j], v[i][j], j, i});
    sort(opts.begin(), opts.end(), [](const Opt& A, const Opt& B) {
        if (A.d != B.d) return A.d > B.d;
        return A.val > B.val;
    });

    vector<int> a(n + 1, 0), rem(cap);
    vector<char> done(n + 1, 0);
    for (auto& o : opts) {
        if (done[o.j]) continue;
        if (w[o.i][o.j] <= rem[o.i]) {
            a[o.j] = o.i; rem[o.i] -= w[o.i][o.j]; done[o.j] = 1;
        }
    }

    for (int j = 1; j <= n; j++) printf("%d%c", a[j], j == n ? '\n' : ' ');
    return 0;
}
