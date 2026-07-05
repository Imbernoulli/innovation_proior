// TIER: greedy
// Cheapest-solo greedy: complete the M orders with the smallest solo round-trip cost,
// one cut at a time, in ascending cost order. Drops the expensive/far orders but never
// batches -- noticeably better than serving everything, still far from optimal.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll md(ll ax, ll ay, ll bx, ll by) {
    return llabs(ax - bx) + llabs(ay - by);
}

int main() {
    int N, Q, M;
    scanf("%d %d %d", &N, &Q, &M);
    ll X0, Y0; scanf("%lld %lld", &X0, &Y0);
    vector<ll> ax(N + 1), ay(N + 1), bx(N + 1), by(N + 1), q(N + 1);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &ax[i], &ay[i], &bx[i], &by[i], &q[i]);

    vector<pair<ll,int>> solo(N);
    for (int i = 1; i <= N; i++) {
        ll c = md(X0, Y0, ax[i], ay[i]) + md(ax[i], ay[i], bx[i], by[i]) + md(bx[i], by[i], X0, Y0);
        solo[i - 1] = {c, i};
    }
    sort(solo.begin(), solo.end());

    int m = max(1, M);
    if (m > N) m = N;

    printf("%d\n", 2 * m);
    for (int k = 0; k < m; k++) {
        int i = solo[k].second;
        printf("0 %d\n", i);
        printf("1 %d\n", i);
    }
    return 0;
}
