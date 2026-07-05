// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static inline ll cheb(ll ax, ll ay, ll bx, ll by) {
    return max(llabs(ax - bx), llabs(ay - by));
}

int main() {
    int P, Q;
    if (scanf("%d %d", &P, &Q) != 2) return 0;
    ll x0, y0;
    scanf("%lld %lld", &x0, &y0);
    vector<ll> px(P + 1), py(P + 1), dx(P + 1), dy(P + 1), q(P + 1), w(P + 1), f(P + 1);
    for (int i = 1; i <= P; i++)
        scanf("%lld %lld %lld %lld %lld %lld %lld",
              &px[i], &py[i], &dx[i], &dy[i], &q[i], &w[i], &f[i]);

    // Serve a request iff skipping it costs more than its standalone round trip.
    vector<int> serve;
    for (int i = 1; i <= P; i++) {
        ll solo = cheb(x0, y0, px[i], py[i])
                + cheb(px[i], py[i], dx[i], dy[i])
                + cheb(dx[i], dy[i], x0, y0);
        if (w[i] > solo) serve.push_back(i);
    }

    // Each served request: pickup then immediate delivery (zero wilting), in input order.
    printf("%d\n", (int)serve.size() * 2);
    for (int i : serve) {
        printf("0 %d\n", i);
        printf("1 %d\n", i);
    }
    return 0;
}
