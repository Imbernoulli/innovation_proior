// TIER: strong
// Orientation-search greedy with gap-filling and deterministic multi-order restarts.
// For several shape orderings: scan tiles row-major, try every dihedral orientation of
// each piece (largest-first) anchoring its bounding box at the tile, place first that fits;
// then fill leftover tiles with scarce dominoes/monominoes. Keep the best-covering packing.
#include <bits/stdc++.h>
using namespace std;

int H, W, D;
vector<vector<pair<int,int>>> base;
vector<int> supply;
vector<vector<char>> blockedGrid;

vector<pair<int,int>> transformShape(int i, int o) {
    vector<pair<int,int>> v;
    for (auto& p : base[i]) {
        int r = p.first, c = p.second;
        if (o >= 4) c = -c;
        for (int t = 0; t < (o % 4); t++) { int nr = c, nc = -r; r = nr; c = nc; }
        v.push_back({r, c});
    }
    int mr = INT_MAX, mc = INT_MAX;
    for (auto& p : v) { mr = min(mr, p.first); mc = min(mc, p.second); }
    for (auto& p : v) { p.first -= mr; p.second -= mc; }
    return v;
}

// precomputed transformed cell sets: orient[i][o]
vector<array<vector<pair<int,int>>,8>> orient;

struct Result { vector<array<int,4>> out; long long cov; };

Result pack(const vector<int>& order, const vector<int>& fillers) {
    vector<vector<char>> occ = blockedGrid;
    vector<int> rem = supply;
    vector<array<int,4>> out;
    long long cov = 0;

    auto tryPlace = [&](int r, int c, const vector<int>& shapeOrder, bool allOrient) {
        for (int idx : shapeOrder) {
            if (rem[idx] <= 0) continue;
            int omax = allOrient ? 8 : 1;
            for (int o = 0; o < omax; o++) {
                auto& cells = orient[idx][o];
                bool fit = true;
                for (auto& p : cells) {
                    int rr = r + p.first, cc = c + p.second;
                    if (rr < 0 || rr >= H || cc < 0 || cc >= W || occ[rr][cc]) { fit = false; break; }
                }
                if (!fit) continue;
                for (auto& p : cells) occ[r + p.first][c + p.second] = 1;
                rem[idx]--;
                out.push_back({idx, o, r, c});
                cov += (long long)cells.size();
                return true;
            }
        }
        return false;
    };

    // main constructive pass: all orientations, given order
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (!occ[r][c]) tryPlace(r, c, order, true);

    // gap-filling pass with scarce small pieces
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (!occ[r][c]) tryPlace(r, c, fillers, true);

    return {out, cov};
}

int main() {
    scanf("%d %d %d", &H, &W, &D);
    base.resize(D); supply.assign(D, 0);
    for (int i = 0; i < D; i++) {
        int A; scanf("%d %d", &A, &supply[i]);
        for (int j = 0; j < A; j++) { int dr, dc; scanf("%d %d", &dr, &dc); base[i].push_back({dr, dc}); }
    }
    int Q; scanf("%d", &Q);
    blockedGrid.assign(H, vector<char>(W, 0));
    for (int j = 0; j < Q; j++) { int r, c; scanf("%d %d", &r, &c); blockedGrid[r][c] = 1; }

    orient.resize(D);
    for (int i = 0; i < D; i++)
        for (int o = 0; o < 8; o++)
            orient[i][o] = transformShape(i, o);

    // fillers: smallest-area shapes first (domino, monomino, ...)
    vector<int> fillers(D);
    for (int i = 0; i < D; i++) fillers[i] = i;
    sort(fillers.begin(), fillers.end(), [&](int a, int b){ return base[a].size() < base[b].size(); });

    // candidate orderings (deterministic)
    vector<vector<int>> orders;
    {
        vector<int> byArea(D);
        for (int i = 0; i < D; i++) byArea[i] = i;
        sort(byArea.begin(), byArea.end(), [&](int a, int b){ return base[a].size() > base[b].size(); });
        orders.push_back(byArea);
    }
    {
        vector<int> byAreaAsc(D);
        for (int i = 0; i < D; i++) byAreaAsc[i] = i;
        sort(byAreaAsc.begin(), byAreaAsc.end(), [&](int a, int b){ return base[a].size() < base[b].size(); });
        orders.push_back(byAreaAsc);
    }
    {
        // several deterministic pseudo-random orderings
        unsigned seed = 0x5bd1e995u;
        for (int t = 0; t < 5; t++) {
            vector<int> ord(D);
            for (int i = 0; i < D; i++) ord[i] = i;
            for (int i = D - 1; i > 0; i--) {
                seed = seed * 1103515245u + 12345u;
                int j = (int)((seed >> 8) % (unsigned)(i + 1));
                swap(ord[i], ord[j]);
            }
            orders.push_back(ord);
        }
    }

    Result best; best.cov = -1;
    for (auto& ord : orders) {
        Result r = pack(ord, fillers);
        if (r.cov > best.cov) best = move(r);
    }

    printf("%d\n", (int)best.out.size());
    for (auto& a : best.out) printf("%d %d %d %d\n", a[0], a[1], a[2], a[3]);
    return 0;
}
