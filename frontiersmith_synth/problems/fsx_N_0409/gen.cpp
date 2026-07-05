#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Tuned Freight-Yard Sorting generator.
// testId is a difficulty/structure ladder:
//   1 tiny (example scale), 2 small random, 3/4 planted good partition,
//   5/9 trap (bipartite max-cut fights the targets), 6/8 needle (one heavy track),
//   7/10 large dense near the envelope.
// Cars 1..nH are HEAVY (carry conflicts). Cars nH+1..n are LIGHT (tonnage in
// 1..m-1, no conflicts) and act as residue adjusters. Targets t_j are set so
// the round-robin routing tunes ZERO tracks (adversarial), maximizing the
// tuning headroom a good solver can exploit.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int T = atoi(argv[1]);

    // ---- ladder table ----
    // type: 0 random, 1 planted, 2 trap, 3 needle
    ll n; int k, m, type; int dense = 0;
    switch (T) {
        case 1:  n = 12;    k = 3;  m = 5;  type = 0; break;
        case 2:  n = 200;   k = 5;  m = 7;  type = 0; break;
        case 3:  n = 800;   k = 6;  m = 7;  type = 1; break;
        case 4:  n = 2000;  k = 8;  m = 7;  type = 1; break;
        case 5:  n = 3000;  k = 6;  m = 11; type = 2; break;
        case 6:  n = 4000;  k = 10; m = 7;  type = 3; break;
        case 7:  n = 8000;  k = 12; m = 11; type = 0; dense = 1; break;
        case 8:  n = 15000; k = 16; m = 13; type = 3; break;
        case 9:  n = 30000; k = 20; m = 13; type = 2; break;
        default: n = 40000; k = 24; m = 17; type = 0; dense = 1; break; // T==10
    }
    int lam = 8;

    // ---- light vs heavy split ----
    ll nL = max((ll)(2 * k), n / 8);
    if (nL > n - (ll)k) nL = n - k;          // keep at least k heavy cars
    if (nL < 1) nL = 1;
    ll nH = n - nL;
    if (nH < 2) { nH = 2; nL = n - 2; }

    // ---- tonnages ----
    vector<ll> w(n + 1);
    for (ll i = 1; i <= nH; i++) w[i] = rnd.next(1, 1000000);
    for (ll i = nH + 1; i <= n; i++) w[i] = rnd.next((ll)1, (ll)(m - 1)); // light adjusters

    // ---- balance band ----
    ll fl = n / k, ce = (n + k - 1) / k;
    ll slack = max((ll)1, fl / 4);
    ll L = max((ll)1, fl - slack);
    ll U = ce + slack;

    // ---- edges (heavy cars only, indices 1..nH) ----
    ll Ebase = (dense ? 6 : (type == 1 ? 5 : (type == 2 ? 6 : (type == 3 ? 5 : 4)))) * nH;
    ll E = min((ll)250000, max((ll)1, Ebase));

    // structural helpers
    vector<int> plantGroup, side, isNeedle;
    if (type == 1) {                          // planted balanced partition
        plantGroup.assign(nH + 1, 0);
        for (ll i = 1; i <= nH; i++) plantGroup[i] = (int)((i - 1) % k);
        shuffle(plantGroup.begin() + 1, plantGroup.end());
    } else if (type == 2) {                   // two adversarial sides
        side.assign(nH + 1, 0);
        for (ll i = 1; i <= nH; i++) side[i] = rnd.next(0, 1);
    } else if (type == 3) {                   // a heavy needle set
        isNeedle.assign(nH + 1, 0);
        ll s = max((ll)2, nH / k);
        for (ll i = 1; i <= s && i <= nH; i++) isNeedle[i] = 1;
    }

    struct Edge { ll u, v, c; };
    vector<Edge> edges;
    edges.reserve(E);

    auto pick = [&]() -> ll { return rnd.next((ll)1, nH); };

    for (ll e = 0; e < E; e++) {
        ll u, v;
        do { u = pick(); v = pick(); } while (u == v);
        ll c;
        if (type == 1) {
            c = (plantGroup[u] != plantGroup[v]) ? rnd.next(400000, 1000000)
                                                 : rnd.next(1, 2000);
        } else if (type == 2) {
            c = (side[u] != side[v]) ? rnd.next(300000, 1000000)
                                     : rnd.next(1, 3000);
        } else if (type == 3) {
            if (isNeedle[u] != isNeedle[v]) c = rnd.next(500000, 1000000); // needle boundary
            else if (isNeedle[u] && isNeedle[v]) c = rnd.next(1, 500);     // keep needle cohesive
            else c = rnd.next(1, 5000);                                    // noise
        } else {
            c = rnd.next(1, 1000000);
        }
        edges.push_back({u, v, c});
    }

    // ---- targets: force round-robin to tune ZERO tracks ----
    // round-robin: car i -> track ((i-1) mod k) (0-indexed here)
    vector<ll> rrSum(k, 0);
    for (ll i = 1; i <= n; i++) rrSum[(i - 1) % k] += w[i];
    vector<int> t(k);
    for (int j = 0; j < k; j++) {
        int shift = rnd.next(1, m - 1);       // nonzero -> round-robin never tuned
        t[j] = (int)(((rrSum[j] % m) + shift) % m);
    }

    // ---- emit ----
    printf("%lld %d %d %d %lld %lld\n", n, k, m, lam, L, U);
    for (ll i = 1; i <= n; i++) printf("%lld%c", w[i], i == n ? '\n' : ' ');
    for (int j = 0; j < k; j++) printf("%d%c", t[j], j == k - 1 ? '\n' : ' ');
    printf("%lld\n", (ll)edges.size());
    for (auto& ed : edges) printf("%lld %lld %lld\n", ed.u, ed.v, ed.c);
    return 0;
}
