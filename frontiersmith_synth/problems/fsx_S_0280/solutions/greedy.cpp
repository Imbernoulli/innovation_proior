// TIER: greedy
// Serve every job whose stand-alone round trip is cheaper than its penalty,
// then chain the served jobs in nearest-neighbour order of their pickups.
// Each job's pickup is emitted immediately before its delivery (precedence trivially ok).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll DST(ll x1, ll y1, ll x2, ll y2) {
    ll dx = x1 - x2, dy = y1 - y2;
    return (ll)llround(sqrt((double)(dx * dx + dy * dy)));
}

int main() {
    int N; ll SX, SY;
    if (scanf("%d %lld %lld", &N, &SX, &SY) != 3) return 0;
    vector<ll> px(N + 1), py(N + 1), dx(N + 1), dy(N + 1), w(N + 1);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &px[i], &py[i], &dx[i], &dy[i], &w[i]);

    // profitability if served alone
    vector<int> chosen;
    for (int i = 1; i <= N; i++) {
        ll cost = DST(SX, SY, px[i], py[i]) + DST(px[i], py[i], dx[i], dy[i]) + DST(dx[i], dy[i], SX, SY);
        if (cost < w[i]) chosen.push_back(i);
    }

    // nearest-neighbour order over chosen pickups starting from depot
    vector<int> order;
    vector<char> used(chosen.size(), 0);
    ll cx = SX, cy = SY;
    for (size_t step = 0; step < chosen.size(); step++) {
        int best = -1; ll bd = LLONG_MAX;
        for (size_t k = 0; k < chosen.size(); k++) {
            if (used[k]) continue;
            ll d = DST(cx, cy, px[chosen[k]], py[chosen[k]]);
            if (d < bd) { bd = d; best = (int)k; }
        }
        used[best] = 1;
        int j = chosen[best];
        order.push_back(j);
        cx = dx[j]; cy = dy[j];   // after delivering, cart is at delivery point
    }

    printf("%d\n", (int)order.size() * 2);
    for (size_t k = 0; k < order.size(); k++) {
        printf("%d %d ", order[k], -order[k]);
    }
    printf("\n");
    return 0;
}
