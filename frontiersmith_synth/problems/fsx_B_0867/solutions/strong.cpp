// TIER: strong
// Insight (the spec's innovation hook): containment under anisotropy is
// interception in spacetime, not enclosure in space. Any path from the
// ignition to a cell at wind-axis projection p > d must cross the line
// proj == d at some cell whose OWN natural (unobstructed) arrival time is
// <= that target's arrival time -- so instead of enclosing the source in
// every direction, it suffices to Dijkstra the natural spread once, then
// search over candidate downwind cut-lines d (beyond the mandatory standoff
// distance) for the one line that is cheap to fully seal (its reachable
// cross-section, using any rock cells as free anchors) yet protects the
// most value beyond it. The cone is wide near the source and narrows toward
// its downwind tip, so the winning line is usually a short, cheap arc well
// past the standoff -- not a ring hugging the ignition.
//
// As a fallback (when no anisotropy-exploiting cut is affordable -- e.g. the
// terrain has no planted narrow corridor to anchor on), also price out a
// full topological ring at the minimum LEGAL radius (exactly the standoff
// distance D): still cheaper than any larger ring, and a real option when
// the budget happens to cover it. Simulate both candidates with the same
// Dijkstra the checker uses and keep whichever construction actually scores
// higher -- the point is picking the right tool for the instance, not
// worshipping one fixed recipe.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
static const ll INF = (ll)1e15;

int H, W, T, S, D, BUDGET, WX, WY, IY, IX;
vector<string> terrain;
vector<vector<ll>> value;

vector<vector<ll>> arrivalTimes(const vector<vector<char>>& cleared) {
    int CF = max(2, S / 2);
    vector<vector<ll>> arr(H, vector<ll>(W, INF));
    priority_queue<pair<ll,int>, vector<pair<ll,int>>, greater<>> pq;
    arr[IY][IX] = 0;
    pq.push({0, IY * W + IX});
    static const int dy[4] = {-1, 1, 0, 0};
    static const int dx[4] = {0, 0, -1, 1};
    while (!pq.empty()) {
        auto [d, node] = pq.top(); pq.pop();
        int uy = node / W, ux = node % W;
        if (d > arr[uy][ux]) continue;
        for (int k = 0; k < 4; k++) {
            int vy = uy + dy[k], vx = ux + dx[k];
            if (vy < 0 || vy >= H || vx < 0 || vx >= W) continue;
            if (terrain[vy][vx] == 'R') continue;
            if (cleared[vy][vx]) continue;
            int dot = (vx - ux) * WX + (vy - uy) * WY;
            int factor = (dot > 0) ? 1 : (dot == 0 ? CF : S);
            int base = (terrain[vy][vx] == 'B') ? 5 : 2;
            ll nd = d + (ll)base * factor;
            if (nd < arr[vy][vx]) { arr[vy][vx] = nd; pq.push({nd, vy * W + vx}); }
        }
    }
    return arr;
}

ll objectiveOf(const vector<vector<char>>& cleared) {
    auto arr = arrivalTimes(cleared);
    ll F = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            if (terrain[y][x] == 'R') continue;
            if (cleared[y][x]) continue;
            if (arr[y][x] <= T) continue;
            F += value[y][x];
        }
    return F;
}

int main() {
    if (scanf("%d %d %d %d %d %d %d %d %d %d", &H, &W, &T, &S, &D, &BUDGET, &WX, &WY, &IY, &IX) != 10) return 0;
    terrain.assign(H, "");
    for (int y = 0; y < H; y++) { char buf[4100]; scanf("%s", buf); terrain[y] = buf; }
    value.assign(H, vector<ll>(W, 0));
    for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) scanf("%lld", &value[y][x]);

    vector<vector<char>> none(H, vector<char>(W, 0)); // no clearing
    auto arr = arrivalTimes(none);

    // bucket cells by wind-axis projection relative to the ignition
    int OFFSET = H + W + 5;
    int maxIdx = 2 * OFFSET + 5;
    vector<vector<pair<int,int>>> bucket(maxIdx);
    int maxProj = 0;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (terrain[y][x] == 'R') continue;
            int proj = (x - IX) * WX + (y - IY) * WY;
            bucket[proj + OFFSET].push_back({y, x});
            if (proj > maxProj) maxProj = proj;
        }
    }

    // Candidate A: the anisotropy-exploiting cut-line search, d in (D, maxProj].
    ll bestVal = -1;
    vector<pair<int,int>> cutCells;
    for (int d = D + 1; d <= maxProj; d++) {
        vector<pair<int,int>> cells;
        for (auto& c : bucket[d + OFFSET]) if (arr[c.first][c.second] <= T) cells.push_back(c);
        if ((int)cells.size() > BUDGET) continue;
        ll beyond = 0;
        for (int dd = d + 1; dd <= maxProj; dd++)
            for (auto& c : bucket[dd + OFFSET])
                if (arr[c.first][c.second] <= T) beyond += value[c.first][c.second];
        if (beyond > bestVal) { bestVal = beyond; cutCells = cells; }
    }
    vector<vector<char>> clearedA(H, vector<char>(W, 0));
    for (auto& c : cutCells) clearedA[c.first][c.second] = 1;
    ll fA = objectiveOf(clearedA);

    // Candidate B: full topological ring at the minimum legal radius D (a
    // guaranteed-safe fallback when no cheap cut-line exists, e.g. no
    // planted corridor to anchor on).
    vector<pair<int,int>> ringCells;
    if (D >= 1) {
        for (int dx = -D; dx <= D; dx++) {
            int dy = D - abs(dx);
            int y1 = IY + dy, x1 = IX + dx;
            if (y1 >= 0 && y1 < H && x1 >= 0 && x1 < W && terrain[y1][x1] != 'R') ringCells.push_back({y1, x1});
            if (dy != 0) {
                int y2 = IY - dy, x2 = IX + dx;
                if (y2 >= 0 && y2 < H && x2 >= 0 && x2 < W && terrain[y2][x2] != 'R') ringCells.push_back({y2, x2});
            }
        }
    }
    ll fB = -1;
    vector<vector<char>> clearedB;
    if ((int)ringCells.size() <= BUDGET) {
        clearedB.assign(H, vector<char>(W, 0));
        for (auto& c : ringCells) clearedB[c.first][c.second] = 1;
        fB = objectiveOf(clearedB);
    }

    const vector<pair<int,int>>* chosen = &cutCells;
    if (fB > fA) chosen = &ringCells;

    printf("%d\n", (int)chosen->size());
    for (auto& c : *chosen) printf("%d %d\n", c.first, c.second);
    return 0;
}
