// TIER: greedy
// The obvious approach: open sites cheapest-cost-first while the budget lasts
// (a plain knapsack-by-cost pass, ignoring capacity/location VALUE), then route
// each demand node (in input order) to the nearest open site that still has
// room, falling back to the next-nearest only when the closest one is already
// full (a bare-minimum overflow check -- it never reconsiders past decisions
// and never weighs the MARGINAL queueing cost of piling on). This still
// concentrates load on cheap, central sites and can drive them to (or past)
// saturation, and can bury a single high-value site under a pile of cheap junk.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(){
    int n, m; ll Bud, QW;
    scanf("%d %d %lld %lld", &n, &m, &Bud, &QW);
    vector<ll> dx(n+1), dy(n+1), dlam(n+1);
    for (int i = 1; i <= n; i++) scanf("%lld %lld %lld", &dx[i], &dy[i], &dlam[i]);
    vector<ll> fx(m+1), fy(m+1), fmu(m+1), fcost(m+1);
    for (int j = 1; j <= m; j++) scanf("%lld %lld %lld %lld", &fx[j], &fy[j], &fmu[j], &fcost[j]);

    vector<int> order(m);
    for (int j = 0; j < m; j++) order[j] = j + 1;
    sort(order.begin(), order.end(), [&](int a, int b){
        if (fcost[a] != fcost[b]) return fcost[a] < fcost[b];
        return a < b;
    });

    vector<int> open;
    ll spent = 0;
    for (int j : order) {
        if (spent + fcost[j] <= Bud) { spent += fcost[j]; open.push_back(j); }
    }
    if (open.empty()) open.push_back(order[0]);   // must open at least one site

    auto dist = [&](int i, int f) {
        double dxv = (double)(dx[i] - fx[f]), dyv = (double)(dy[i] - fy[f]);
        return sqrt(dxv * dxv + dyv * dyv);
    };

    vector<ll> load(m + 1, 0);
    vector<int> assign(n + 1);
    for (int i = 1; i <= n; i++) {
        // sites nearest-first for this demand node
        vector<int> byDist = open;
        sort(byDist.begin(), byDist.end(), [&](int a, int b){ return dist(i, a) < dist(i, b); });
        int chosen = -1;
        for (int f : byDist) {
            if (load[f] + dlam[i] < fmu[f]) { chosen = f; break; }   // first with room
        }
        if (chosen == -1) chosen = byDist[0];   // no room anywhere: best-effort (may saturate)
        load[chosen] += dlam[i];
        assign[i] = chosen;
    }

    printf("%d\n", (int)open.size());
    for (int f : open) printf("%d ", f);
    printf("\n");
    for (int i = 1; i <= n; i++) printf("%d ", assign[i]);
    printf("\n");
    return 0;
}
