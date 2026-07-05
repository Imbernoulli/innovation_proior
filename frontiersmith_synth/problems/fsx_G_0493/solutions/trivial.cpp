// TIER: trivial
// One operation per cycle in program (topological) order, spaced to satisfy latencies.
// This is EXACTLY the grader's baseline B construction, so it scores ratio ~= 0.1.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int n, m, W, T, P;
    if (scanf("%d %d %d %d %d", &n, &m, &W, &T, &P) != 5) return 0;
    vector<int> cap(T + 1), rp(T + 1);
    for (int t = 1; t <= T; t++) scanf("%d", &cap[t]);
    for (int t = 1; t <= T; t++) scanf("%d", &rp[t]);
    vector<int> type(n + 1), L(n + 1);
    for (int i = 1; i <= n; i++) scanf("%d %d", &type[i], &L[i]);
    vector<vector<int>> preds(n + 1);
    for (int j = 0; j < m; j++) {
        int u, v; scanf("%d %d", &u, &v);
        preds[v].push_back(u);
    }
    vector<long long> s(n + 1, 0);
    long long cur = -1;
    for (int i = 1; i <= n; i++) {
        long long e = 0;
        for (int u : preds[i]) e = max(e, s[u] + (long long)L[u]);
        s[i] = max(e, cur + 1);
        cur = s[i];
    }
    for (int i = 1; i <= n; i++) printf("%lld\n", s[i]);
    return 0;
}
