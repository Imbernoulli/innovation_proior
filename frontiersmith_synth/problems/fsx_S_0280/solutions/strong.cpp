// TIER: strong
// Prize-collecting cheapest-insertion. Repeatedly insert the pickup+delivery pair
// (as two adjacent events) at the route position with least marginal driving cost,
// as long as that marginal cost is below the penalty it saves. Then a light 2-opt
// style relocate pass on served pairs to trim the route further.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

static ll DST(ll x1, ll y1, ll x2, ll y2) {
    ll dx = x1 - x2, dy = y1 - y2;
    return (ll)llround(sqrt((double)(dx * dx + dy * dy)));
}

int N; ll SX, SY;
vector<ll> px, py, dx, dy, w;

// coordinate of a signed event token
static inline void coordOf(int v, ll &x, ll &y) {
    int j = abs(v);
    if (v > 0) { x = px[j]; y = py[j]; } else { x = dx[j]; y = dy[j]; }
}

int main() {
    if (scanf("%d %lld %lld", &N, &SX, &SY) != 3) return 0;
    px.assign(N + 1, 0); py.assign(N + 1, 0); dx.assign(N + 1, 0); dy.assign(N + 1, 0); w.assign(N + 1, 0);
    for (int i = 1; i <= N; i++)
        scanf("%lld %lld %lld %lld %lld", &px[i], &py[i], &dx[i], &dy[i], &w[i]);

    vector<int> route;            // sequence of signed events (between depot and depot)
    vector<char> served(N + 1, 0);

    auto nodeCoord = [&](int idx, ll &x, ll &y) {
        // idx in [0, route.size()] over the full path depot,route...,depot
        if (idx == 0) { x = SX; y = SY; }
        else if (idx == (int)route.size() + 1) { x = SX; y = SY; }
        else coordOf(route[idx - 1], x, y);
    };

    while (true) {
        int m = (int)route.size();
        // precompute full-path node coords (0..m+1)
        ll bestProfit = 0;        // require strictly positive to insert
        int bestJob = -1, bestGap = -1;
        for (int j = 1; j <= N; j++) {
            if (served[j]) continue;
            ll pjx = px[j], pjy = py[j], djx = dx[j], djy = dy[j];
            ll ppd = DST(pjx, pjy, djx, djy);
            // best gap to insert pair (+j,-j)
            ll bestDelta = LLONG_MAX; int bestG = -1;
            for (int g = 0; g <= m; g++) {
                ll lx, ly, rx, ry;
                nodeCoord(g, lx, ly);
                nodeCoord(g + 1, rx, ry);
                ll delta = DST(lx, ly, pjx, pjy) + ppd + DST(djx, djy, rx, ry) - DST(lx, ly, rx, ry);
                if (delta < bestDelta) { bestDelta = delta; bestG = g; }
            }
            ll profit = w[j] - bestDelta;
            if (profit > bestProfit) { bestProfit = profit; bestJob = j; bestGap = bestG; }
        }
        if (bestJob < 0) break;
        // insert pair at gap bestGap: route positions [bestGap] gets +j, -j
        route.insert(route.begin() + bestGap, -bestJob);
        route.insert(route.begin() + bestGap, bestJob);
        served[bestJob] = 1;
    }

    // Light relocate pass: try moving each served pair to a cheaper insertion spot.
    for (int pass = 0; pass < 2; pass++) {
        bool improved = false;
        for (int j = 1; j <= N; j++) {
            if (!served[j]) continue;
            // remove pair (+j,-j) from route
            int ip = -1, id = -1;
            for (int k = 0; k < (int)route.size(); k++) {
                if (route[k] == j) ip = k;
                else if (route[k] == -j) id = k;
            }
            if (ip < 0 || id < 0) continue;
            // remove larger index first
            int hi = max(ip, id), lo = min(ip, id);
            route.erase(route.begin() + hi);
            route.erase(route.begin() + lo);
            int m = (int)route.size();
            ll pjx = px[j], pjy = py[j], djx = dx[j], djy = dy[j];
            ll ppd = DST(pjx, pjy, djx, djy);
            ll bestDelta = LLONG_MAX; int bestG = -1;
            for (int g = 0; g <= m; g++) {
                ll lx, ly, rx, ry;
                nodeCoord(g, lx, ly);
                nodeCoord(g + 1, rx, ry);
                ll delta = DST(lx, ly, pjx, pjy) + ppd + DST(djx, djy, rx, ry) - DST(lx, ly, rx, ry);
                if (delta < bestDelta) { bestDelta = delta; bestG = g; }
            }
            // reinsert at best spot (may be same place)
            route.insert(route.begin() + bestG, -j);
            route.insert(route.begin() + bestG, j);
            // note: relocate only rearranges; served set unchanged
            (void)improved;
        }
    }

    printf("%d\n", (int)route.size());
    for (size_t k = 0; k < route.size(); k++) printf("%d ", route[k]);
    printf("\n");
    return 0;
}
