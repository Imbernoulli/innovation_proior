// TIER: trivial
// First-fit in input order -> exactly reproduces the checker's baseline B (ratio 0.1).
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
    vector<int> a(n + 1, 0);
    for (int j = 1; j <= n; j++) {
        for (int i = 1; i <= m; i++) {
            if (rem[i] >= e[i][j]) { rem[i] -= e[i][j]; a[j] = i; break; }
        }
    }
    for (int j = 1; j <= n; j++) printf("%d%c", a[j], j == n ? '\n' : ' ');
    return 0;
}
