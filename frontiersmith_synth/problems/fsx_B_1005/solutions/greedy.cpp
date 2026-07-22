// TIER: greedy
#include <bits/stdc++.h>
using namespace std;

// The obvious first instinct: heat shed scales with exposed face count, so
// greedily grow the shape by always adding whichever reachable, unused cell
// maximizes the IMMEDIATE exposed-perimeter delta (4 - 2*occupiedNeighbors).
// This never accepts a cell with 2+ occupied neighbors while ANY 1-neighbor
// cell is still reachable, so the shape never gets wider than one block
// anywhere -- it spends its whole budget on unplanned one-cell-wide arms and
// never reasons about conduction at all.

static inline unsigned long long mix(long long r, long long c) {
    unsigned long long x = (unsigned long long) (r * 1000003LL + c * 998244353LL);
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int main() {
    int H, W, M, Thot, Tamb, Kcond, Kloss, BW, BC;
    cin >> H >> W >> M >> Thot >> Tamb >> Kcond >> Kloss >> BW >> BC;

    vector<vector<char>> occ(H + 1, vector<char>(W, 0));
    auto isBaseNbr = [&](int r, int c) { return r == 1 && c >= BC && c < BC + BW; };
    // k = occupied-or-wall neighbor count, used for the exposed-perimeter delta
    // (matches the checker's own exposed(i) = 4 - k exactly).
    auto occupiedNbrCount = [&](int r, int c) {
        int k = 0;
        int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr == 0) { if (isBaseNbr(r, c) && dr[d] == -1) k++; continue; }
            if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
            if (occ[nr][nc]) k++;
        }
        return k;
    };
    // Whether a candidate actually joins the SINGLE growing structure: adjacent
    // to an already-PLACED block, or (only while nothing has been placed yet)
    // adjacent to the wall. Wall-adjacency alone is NOT enough once the shape
    // already has a root, otherwise a wide wall would let the greedy spawn a
    // second, disconnected trunk from a different wall column.
    auto joinsStructure = [&](int r, int c, bool haveRoot) {
        int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
        for (int d = 0; d < 4; d++) {
            int nr = r + dr[d], nc = c + dc[d];
            if (nr < 1 || nr > H || nc < 0 || nc >= W) continue;
            if (occ[nr][nc]) return true;
        }
        return !haveRoot && isBaseNbr(r, c);
    };

    vector<pair<int,int>> chosen;
    for (int step = 0; step < M; step++) {
        bool haveRoot = !chosen.empty();
        int bestDelta = INT_MIN; unsigned long long bestHash = ULLONG_MAX;
        int br = -1, bc = -1;
        for (int r = 1; r <= H; r++) {
            for (int c = 0; c < W; c++) {
                if (occ[r][c]) continue;
                if (!joinsStructure(r, c, haveRoot)) continue;
                int k = occupiedNbrCount(r, c);
                int delta = 4 - 2 * k;
                unsigned long long h = mix(r, c);
                if (delta > bestDelta || (delta == bestDelta && h < bestHash)) {
                    bestDelta = delta; bestHash = h; br = r; bc = c;
                }
            }
        }
        if (br == -1) break;
        occ[br][bc] = 1;
        chosen.push_back({br, bc});
    }

    cout << chosen.size() << "\n";
    for (auto &pr : chosen) cout << pr.first << " " << pr.second << "\n";
    return 0;
}
