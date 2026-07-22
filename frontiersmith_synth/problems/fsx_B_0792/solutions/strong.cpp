// TIER: strong
// Insight: barriers are strongest against the flow's WEAK axis. Instead of
// fighting each packet's full momentum head-on (a wall, expensive and capped
// at HMAX), steer it: a single oblique DEFLECT barrier only needs to absorb
// the turn-1 fraction of the momentum (~40%), which fits under HMAX across
// the whole planted momentum range, and a successful deflection also
// dissipates a further chunk of momentum, so the packet drifts toward open,
// low-weight ground with reduced impact even where it eventually lands.
// Exchange-argument prioritization: process releases in decreasing order of
// their UNMITIGATED impact (w[c]*m*p) so the budget -- never quite unlimited
// -- is spent on the biggest threats first. Each release gets exactly one
// early, cheap deflector (row 1) turning it away from the village's peak
// weight column; since release columns are pairwise distinct, no two
// deflectors ever collide.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static const int HMAX = 100;
static const int FRAC1 = 40;

int main() {
    int H, W, K; ll L; int SCALE, DISS;
    if (scanf("%d %d %d %lld %d %d", &H, &W, &K, &L, &SCALE, &DISS) != 6) return 0;
    vector<int> w(W);
    int peak = 0;
    for (int c = 0; c < W; c++) { scanf("%d", &w[c]); if (w[c] > w[peak]) peak = c; }
    vector<ll> c(K), m(K), p(K);
    for (int i = 0; i < K; i++) scanf("%lld %lld %lld", &c[i], &m[i], &p[i]);
    (void)DISS;

    vector<int> order(K);
    for (int i = 0; i < K; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) {
        return (ll)w[c[a]] * m[a] * p[a] > (ll)w[c[b]] * m[b] * p[b];
    });

    vector<tuple<int,int,int,int>> placed;
    ll spent = 0;
    for (int idx : order) {
        ll normal = (p[idx] * (ll)FRAC1 + 99) / 100;
        ll hNeeded = (normal + SCALE - 1) / SCALE; // ceil(normal / SCALE)
        if (hNeeded < 1) hNeeded = 1;
        if (hNeeded > HMAX) continue;              // this one hop can't afford it -- skip
        if (spent + hNeeded > L) continue;          // out of budget -- skip

        int target = (c[idx] < peak) ? 1 /*LEFT*/ : 2 /*RIGHT*/; // orientation: away from the peak
        int row = (H > 1) ? 1 : 0;
        placed.push_back({row, (int)c[idx], target, (int)hNeeded});
        spent += hNeeded;
    }

    printf("%d\n", (int)placed.size());
    for (auto& t : placed) {
        printf("%d %d %d %d\n", get<0>(t), get<1>(t), get<2>(t), get<3>(t));
    }
    return 0;
}
