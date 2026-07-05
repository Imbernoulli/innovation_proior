// TIER: trivial
// Best affordable cable-connected pair -> F == B -> ratio 0.1.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int N, M; ll C;
    if (scanf("%d %d %lld", &N, &M, &C) != 3) return 0;
    vector<ll> p(N + 1), c(N + 1);
    for (int i = 1; i <= N; i++) scanf("%lld %lld", &p[i], &c[i]);
    map<pair<int,int>, ll> pairW;
    for (int j = 0; j < M; j++) {
        int u, v; ll w; scanf("%d %d %lld", &u, &v, &w);
        int a = min(u, v), b = max(u, v);
        pairW[{a, b}] += w;
    }
    ll best = -1; int bu = -1, bv = -1;
    for (auto& kv : pairW) {
        int u = kv.first.first, v = kv.first.second;
        if (c[u] + c[v] <= C) {
            ll cand = kv.second * (p[u] + p[v]);
            if (cand > best) { best = cand; bu = u; bv = v; }
        }
    }
    if (bu < 0) { printf("0\n"); return 0; }
    printf("2\n%d %d\n", bu, bv);
    return 0;
}
