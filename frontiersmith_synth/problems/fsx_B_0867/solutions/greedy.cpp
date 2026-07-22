// TIER: greedy
// The obvious first instinct: "seal the fire in where it started." Build a
// ring of cleared cells at exactly the legal standoff distance D around the
// ignition point, enumerated in a fixed, WIND-OBLIVIOUS raster order (by
// column, scanning the grid's own +x to -x direction, ties broken by row) --
// spending the budget cell by cell (skipping cells already rock, or
// off-grid) until it runs out. This never even LOOKS at the wind vector, so
// on every test where the fire's real cone points toward -x (west) it never
// reaches those cells before the budget runs out -- the ring is left open on
// exactly the side that matters, and the cells it did build (upwind/
// crosswind of the ignition, where the fire barely spreads within the
// horizon anyway) buy essentially nothing.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int H, W, T, S, D, BUDGET, WX, WY, IY, IX;
    if (scanf("%d %d %d %d %d %d %d %d %d %d", &H, &W, &T, &S, &D, &BUDGET, &WX, &WY, &IY, &IX) != 10) return 0;
    (void)T; (void)S; (void)WX; (void)WY;
    vector<string> terrain(H);
    for (int y = 0; y < H; y++) { char buf[4100]; scanf("%s", buf); terrain[y] = buf; }
    // value grid must be consumed even though greedy ignores it
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) { long long v; scanf("%lld", &v); }

    vector<pair<int,int>> ring; // (y, x)
    if (D >= 1) {
        for (int dx = -D; dx <= D; dx++) {
            int dy = D - abs(dx);
            ring.push_back({IY + dy, IX + dx});
            if (dy != 0) ring.push_back({IY - dy, IX + dx});
        }
    }
    // Enumerate strictly by column, HIGH x (east) first, LOW x (west) last --
    // a fixed grid-scan convention that never consults WX/WY.
    sort(ring.begin(), ring.end(), [](const pair<int,int>& a, const pair<int,int>& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    vector<pair<int,int>> chosen;
    int budget = BUDGET;
    for (auto& c : ring) {
        if (budget <= 0) break;
        int y = c.first, x = c.second;
        if (y < 0 || y >= H || x < 0 || x >= W) continue;
        if (terrain[y][x] == 'R') continue; // already blocks, free
        if (y == IY && x == IX) continue;   // never true at distance D>=1, defensive
        chosen.push_back({y, x});
        budget--;
    }

    printf("%d\n", (int)chosen.size());
    for (auto& c : chosen) printf("%d %d\n", c.first, c.second);
    return 0;
}
