// TIER: trivial
// Reproduces the checker's first-fit reference assignment exactly -> F == B -> ratio 0.1.
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

    vector<int> a(n + 1, 0), rem(cap);
    vector<char> used(n + 1, 0);
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (!used[j] && w[i][j] <= rem[i]) { used[j] = 1; rem[i] -= w[i][j]; a[j] = i; }

    for (int j = 1; j <= n; j++) printf("%d%c", a[j], j == n ? '\n' : ' ');
    return 0;
}
