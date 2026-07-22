// TIER: greedy
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Obvious "accretive growth" mental model: the coral is ONE growing tip that
// keeps adding to itself, one layer at a time -- so build a SINGLE connected
// dendrite from bedrock to the top, greedily picking (within the allowed
// overhang S of the current tip) whichever next cell looks locally best right
// now, with no lookahead. This is a perfectly legal, single-pass heuristic and
// it never violates the support rule -- but it never realizes that ONE buttress
// cell can support MANY simultaneous branch tips in the row above it, so once
// the single dendrite reaches the top of the grid it just stops, leaving most of
// a large mass budget M unused (it doesn't know how to spend it).

static ll W, H, S, M, Fx, Fy;
static const ll BASE = 5;

ll bonus(ll dx, ll dy) { ll v = Fx * dx + Fy * dy; return v > 0 ? v : 0; }

int main() {
    cin >> W >> H >> S >> M >> Fx >> Fy;

    vector<vector<char>> occ(W + 2, vector<char>(H + 2, 0));
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

    // seed the dendrite: try middle column first, else the first column that works
    ll x0 = (W + 1) / 2;
    occ[x0][1] = 1;
    placed.push_back({x0, 1});
    ll curX = x0, curY = 1;

    while ((ll)placed.size() < M && curY < H) {
        ll lo = max((ll)1, curX - S), hi = min(W, curX + S);
        ll bestX = -1, bestGain = -1;
        for (ll xp = lo; xp <= hi; xp++) {
            if (occ[xp][curY + 1]) continue;
            ll g = gain(xp, curY + 1);
            if (g > bestGain || (g == bestGain && xp < bestX)) {
                bestGain = g; bestX = xp;
            }
        }
        if (bestX < 0) break; // boxed in (shouldn't happen given S>=1)
        curY += 1; curX = bestX;
        occ[curX][curY] = 1;
        placed.push_back({curX, curY});
    }

    printf("%lld\n", (ll)placed.size());
    for (auto &p : placed) printf("%lld %lld\n", p.first, p.second);
    return 0;
}
