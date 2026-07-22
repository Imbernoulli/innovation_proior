// TIER: strong
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Insight: a single connected trunk cell can act as a buttress that supports
// MANY simultaneous branch tips in the row above (everything within +-S of it),
// so instead of growing one dendrite at a time (which wastes almost all of a
// large mass budget once it reaches the ceiling), keep a full frontier of every
// currently-growable cell and, at each step, place the cell whose OWN four faces
// are worth the most right now (own newly-exposed faces, current-alignment bonus
// included) -- this naturally spaces branches apart (an isolated branch tip earns
// far more than one crammed against existing material, since crowding covers
// faces) and lets a small buttress fan out into a whole current-facing canopy
// that spends the full mass budget instead of stopping at the first tall dendrite.

static ll W, H, S, M, Fx, Fy;
static const ll BASE = 5;

ll bonus(ll dx, ll dy) { ll v = Fx * dx + Fy * dy; return v > 0 ? v : 0; }

int main() {
    cin >> W >> H >> S >> M >> Fx >> Fy;

    vector<vector<char>> occ(W + 2, vector<char>(H + 2, 0));
    vector<vector<char>> isCand(W + 2, vector<char>(H + 2, 0));
    vector<pair<ll,ll>> candList;

    for (ll x = 1; x <= W; x++) { isCand[x][1] = 1; candList.push_back({x, 1}); }

    ll dxs[4] = {-1, 1, 0, 0};
    ll dys[4] = {0, 0, 1, -1};

    auto gain = [&](ll x, ll y) -> ll {
        ll g = 0;
        for (int d = 0; d < 4; d++) {
            ll nx = x + dxs[d], ny = y + dys[d];
            bool solid;
            if (ny == 0) solid = true;
            else if (nx < 1 || nx > W || ny > H) solid = false;
            else solid = occ[nx][ny];
            if (!solid) g += BASE + bonus(dxs[d], dys[d]);
        }
        return g;
    };

    vector<pair<ll,ll>> placed;
    ll placedCount = 0;

    while (placedCount < M) {
        ll bestIdx = -1, bestGain = -1;
        ll bestX = -1, bestY = -1;
        for (size_t i = 0; i < candList.size(); i++) {
            ll x = candList[i].first, y = candList[i].second;
            if (occ[x][y]) continue; // lazily-stale entry
            ll g = gain(x, y);
            if (g > bestGain || (g == bestGain && (y < bestY || (y == bestY && x < bestX)))) {
                bestGain = g; bestIdx = (ll)i; bestX = x; bestY = y;
            }
        }
        if (bestIdx < 0) break; // no more growable cells
        occ[bestX][bestY] = 1;
        placed.push_back({bestX, bestY});
        placedCount++;
        // newly growable cells directly above, within S
        if (bestY + 1 <= H) {
            ll lo = max((ll)1, bestX - S), hi = min(W, bestX + S);
            for (ll xp = lo; xp <= hi; xp++) {
                if (!occ[xp][bestY + 1] && !isCand[xp][bestY + 1]) {
                    isCand[xp][bestY + 1] = 1;
                    candList.push_back({xp, bestY + 1});
                }
            }
        }
    }

    printf("%lld\n", (ll)placed.size());
    for (auto &p : placed) printf("%lld %lld\n", p.first, p.second);
    return 0;
}
