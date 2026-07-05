#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// ---- Glacier Sensor Net: min-cost geometric covering-facility-location ----
// testId is a difficulty/structure ladder:
//   testId 1  : tiny (example scale), uniform demand
//   growing side length + demand count; even testIds cluster the demand
//   (skewed / crevasse-field structure), odd testIds are near-uniform.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    const ll W = 20000; // glacier extent (metres), coords in [0, W]

    // ---- lattice of "backbone" beacons guaranteeing full coverage ----
    int g = 4 + t;                       // beacons per side: 5 .. 14
    ll s = W / (g - 1);                  // nominal spacing
    ll Rg = (ll)ceil(s * 1.0) + 2;       // backbone radius: guarantees the box is covered

    int nExtra = (int)llround(0.6 * (double)g * g); // local sensors (fewer than backbone)
    int nd = (int)llround(1.5 * (double)g * g);     // demand stations

    // ---- demand stations ----
    vector<pair<ll,ll>> dem;
    dem.reserve(nd);
    bool clustered = (t % 2 == 0);
    int nClusters = 3 + (t % 3);
    vector<pair<ll,ll>> ctr;
    for (int c = 0; c < nClusters; c++)
        ctr.push_back({rnd.next(0LL, W), rnd.next(0LL, W)});
    ll spread = max((ll)1, s); // cluster spread ~ one cell
    for (int i = 0; i < nd; i++) {
        ll x, y;
        if (clustered && rnd.next(0, 99) < 72) {
            auto& c = ctr[rnd.next(0, (int)ctr.size() - 1)];
            x = min(W, max(0LL, c.first  + rnd.next(-spread, spread)));
            y = min(W, max(0LL, c.second + rnd.next(-spread, spread)));
        } else {
            x = rnd.next(0LL, W);
            y = rnd.next(0LL, W);
        }
        dem.push_back({x, y});
    }

    // ---- sensors: backbone lattice + local candidates ----
    // sensor tuple: sx sy radius cost
    vector<array<ll,4>> sen;
    auto costOf = [&](ll r) -> ll {
        ll base = 30 + (ll)llround((double)r * (double)r / 300000.0);
        return base + rnd.next(0LL, 40LL);
    };
    // backbone lattice (expensive, wide) -> guarantees "all-on" is feasible
    for (int i = 0; i < g; i++)
        for (int j = 0; j < g; j++) {
            ll x = (ll)i * W / (g - 1);
            ll y = (ll)j * W / (g - 1);
            sen.push_back({x, y, Rg, costOf(Rg)});
        }
    // local candidates (cheaper, narrower) -> cost-effective where demand is dense
    for (int k = 0; k < nExtra; k++) {
        ll x = rnd.next(0LL, W);
        ll y = rnd.next(0LL, W);
        ll r = rnd.next(s / 3, (ll)llround(1.1 * s));
        if (r < 1) r = 1;
        sen.push_back({x, y, r, costOf(r)});
    }
    // shuffle sensors so backbone is not trivially the prefix
    for (int i = (int)sen.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(sen[i], sen[j]);
    }

    int N = (int)dem.size();
    int M = (int)sen.size();
    printf("%d %d\n", N, M);
    for (auto& d : dem) printf("%lld %lld\n", d.first, d.second);
    for (auto& e : sen) printf("%lld %lld %lld %lld\n", e[0], e[1], e[2], e[3]);
    return 0;
}
