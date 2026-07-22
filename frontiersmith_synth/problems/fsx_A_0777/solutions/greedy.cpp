// TIER: greedy
// The "obvious" first instinct: weave a local mesh. Connect villages at
// index-distance w=1, then w=2, then w=3, ... (nearest villages first),
// respecting the degree cap and the wirelength budget, until nothing more
// fits. This is a standard, very natural recipe for "a fast local network
// under a budget" -- it never reasons about the ridge's fold structure at
// all, so once the degree cap is used up on cheap local links it can never
// reach a fold-crossing edge, and the network stays locally dense but
// globally thin (large hop-diameter across the ridge).
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int n, k, d; ll W;
    if (!(cin >> n >> k >> d >> W)) return 0;
    vector<ll> x(n);
    for (auto &v : x) cin >> v;

    vector<int> deg(n, 0);
    ll total = 0;
    vector<pair<int,int>> edges;

    for (int w = 1; w < n; w++) {
        bool addedAny = false;
        for (int i = 0; i + w < n; i++) {
            int j = i + w;
            if (deg[i] >= d || deg[j] >= d) continue;
            ll cost = x[j] - x[i];
            if (total + cost > W) continue;
            deg[i]++; deg[j]++;
            total += cost;
            edges.push_back({i, j});
            addedAny = true;
        }
        if (!addedAny) {
            // nothing at this bandwidth fit anywhere; wider bandwidths are
            // only more expensive per pair on average, so stop.
            bool anyRoom = false;
            for (int i = 0; i < n; i++) if (deg[i] < d) { anyRoom = true; break; }
            if (!anyRoom) break;
        }
    }

    cout << edges.size() << "\n";
    for (auto &e : edges) cout << (e.first + 1) << ' ' << (e.second + 1) << "\n";
    return 0;
}
