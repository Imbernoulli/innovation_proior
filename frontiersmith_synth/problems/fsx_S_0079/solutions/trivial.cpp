// TIER: trivial
// Daisy-chain baseline: install the cheapest direct cable between each consecutive
// station pair (i, i+1). Feasible (degree <= 2) and equals the checker baseline B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int main() {
    int n, m;
    if (scanf("%d %d", &n, &m) != 2) return 0;
    vector<int> d(n + 1);
    for (int v = 1; v <= n; v++) scanf("%d", &d[v]);
    vector<ll> bestW(n + 1, LLONG_MAX);
    vector<int> bestIdx(n + 1, -1);
    for (int i = 1; i <= m; i++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        int lo = min(u, v), hi = max(u, v);
        if (hi == lo + 1 && w < bestW[lo]) { bestW[lo] = w; bestIdx[lo] = i; }
    }
    vector<int> out;
    for (int i = 1; i < n; i++) if (bestIdx[i] != -1) out.push_back(bestIdx[i]);
    printf("%d\n", (int)out.size());
    for (int e : out) printf("%d\n", e);
    return 0;
}
