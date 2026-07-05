// TIER: trivial
// Naive first-fit schedule -- exactly reproduces the checker's baseline B, so ratio ~= 0.1.
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

    vector<long long> rem = C;
    vector<int> a(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            if (w[i][j] <= rem[j]) { rem[j] -= w[i][j]; a[i] = j; break; }
        }
    }
    for (int i = 1; i <= n; i++) printf("%d%c", a[i], i == n ? '\n' : ' ');
    return 0;
}
