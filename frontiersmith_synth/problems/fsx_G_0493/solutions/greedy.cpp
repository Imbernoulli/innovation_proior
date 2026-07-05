// TIER: greedy
// In-order list scheduling: walk operations in program order, drop each into the earliest
// cycle whose issue width, per-kind unit cap and read-port budget still admit it (and whose
// latency constraints from already-placed predecessors are met).  Packs bundles but is
// easily trapped by program order.
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

    // per-cycle counters
    unordered_map<long long, int> cntCycle;
    unordered_map<long long, long long> portCycle;
    unordered_map<long long, int> cntType;   // cycle*(T+1)+kind
    cntCycle.reserve(2 * n); portCycle.reserve(2 * n); cntType.reserve(2 * n);

    vector<long long> s(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long e = 0;
        for (int u : preds[i]) e = max(e, s[u] + (long long)L[u]);
        long long c = e;
        while (true) {
            int cc = cntCycle.count(c) ? cntCycle[c] : 0;
            long long kt = c * (long long)(T + 1) + type[i];
            int ct = cntType.count(kt) ? cntType[kt] : 0;
            long long pv = portCycle.count(c) ? portCycle[c] : 0;
            if (cc < W && ct < cap[type[i]] && pv + rp[type[i]] <= P) break;
            c++;
        }
        s[i] = c;
        cntCycle[c] += 1;
        cntType[c * (long long)(T + 1) + type[i]] += 1;
        portCycle[c] += rp[type[i]];
    }
    for (int i = 1; i <= n; i++) printf("%lld\n", s[i]);
    return 0;
}
