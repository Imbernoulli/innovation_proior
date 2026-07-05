// TIER: strong
// Geometry-aware: build a nearest-neighbor tour through profile space (so
// geometrically-similar acts become contiguous, near-zero intra-stage setup), then cut
// the tour into M contiguous runtime-balanced segments (one per stage). Balances the
// makespan AND keeps setup small -> beats the geometry-blind load balancer.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int N, M, d;
vector<ll> p;
vector<vector<ll>> X;
static inline ll dist2(int a, int b) {
    ll s = 0; for (int k = 0; k < d; k++) { ll dd = X[a][k] - X[b][k]; s += dd * dd; } return s;
}
int main() {
    if (scanf("%d %d %d", &N, &M, &d) != 3) return 0;
    p.assign(N + 1, 0);
    X.assign(N + 1, vector<ll>(d));
    for (int j = 1; j <= N; j++) { scanf("%lld", &p[j]); for (int k = 0; k < d; k++) scanf("%lld", &X[j][k]); }

    // nearest-neighbor tour in profile space
    vector<char> used(N + 1, 0);
    vector<int> tour; tour.reserve(N);
    int cur = 1; used[1] = 1; tour.push_back(1);
    for (int step = 1; step < N; step++) {
        ll bd = LLONG_MAX; int bj = -1;
        for (int j = 1; j <= N; j++) if (!used[j]) { ll dd = dist2(cur, j); if (dd < bd) { bd = dd; bj = j; } }
        used[bj] = 1; tour.push_back(bj); cur = bj;
    }

    ll T = 0; for (int j = 1; j <= N; j++) T += p[j];
    ll target = (T + M - 1) / M;

    vector<vector<int>> st(M);
    int s = 0; ll cload = 0;
    for (int i = 0; i < N; i++) {
        st[s].push_back(tour[i]);
        cload += p[tour[i]];
        int remainStages = M - 1 - s;
        int remainActs = N - 1 - i;
        if (s < M - 1 && cload >= target && remainActs > remainStages) { s++; cload = 0; }
    }

    for (int i = 0; i < M; i++) {
        printf("%d", (int)st[i].size());
        for (int a : st[i]) printf(" %d", a);
        printf("\n");
    }
    return 0;
}
