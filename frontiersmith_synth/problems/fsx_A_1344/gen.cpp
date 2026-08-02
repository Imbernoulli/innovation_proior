// testlib generator for fsx_A_1344 (Sparse Gates containment).
#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct TestParam {
    int N, T, S, K, B, G;   // grid, turns, speed, per-turn cap, total budget, #gates
    bool decoy;             // plant a near-but-too-wide-to-seal-in-time decoy gate
};

// per-testId ladder: sizes grow, >=5 cases are TRAPS (a nearby gate that is
// structurally impossible to finish sealing before the runner reaches it --
// reactive/nearest-first blocking wastes its whole budget there while a
// feasibility-aware planner spends it sealing the OTHER, farther-but-cheap
// gates instead).
static TestParam paramFor(int id) {
    switch (id) {
        case 1:  return {10, 8, 1, 2, 6, 1, false};
        case 2:  return {12, 9, 1, 2, 7, 2, false};
        case 3:  return {14, 10, 2, 2, 8, 2, false};
        case 4:  return {17, 10, 3, 2, 10, 4, true};   // TRAP
        case 5:  return {19, 12, 3, 3, 12, 4, true};   // TRAP
        case 6:  return {20, 12, 1, 3, 10, 4, false};
        case 7:  return {23, 14, 4, 3, 14, 4, true};   // TRAP
        case 8:  return {25, 14, 2, 3, 12, 4, false};
        case 9:  return {28, 16, 3, 4, 16, 4, true};   // TRAP
        case 10: return {32, 18, 3, 4, 18, 4, true};   // TRAP, largest
        default: return {10, 8, 1, 2, 6, 1, false};
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    TestParam p = paramFor(testId);
    int N = p.N;

    vector<string> grid(N, string(N, '.'));
    for (int c = 0; c < N; c++) { grid[0][c] = '#'; grid[N - 1][c] = '#'; }
    for (int r = 0; r < N; r++) { grid[r][0] = '#'; grid[r][N - 1] = '#'; }

    auto placeRun = [&](int side, int start, int w) {
        for (int pos = start; pos < start + w; pos++) {
            int r, c;
            if (side == 0) { r = 0; c = pos; }
            else if (side == 1) { r = N - 1; c = pos; }
            else if (side == 2) { r = pos; c = 0; }
            else { r = pos; c = N - 1; }
            grid[r][c] = '.';
        }
    };

    int rx = N / 2, ry = N / 2;
    vector<pair<int,int>> allGateCells;

    if (p.decoy) {
        // 1) place the runner at a controlled distance from a chosen side,
        //    2) plant a WIDE decoy gate on that side, centered opposite the
        //       runner, whose build time (ceil(width/K)) exceeds the turns
        //       available before the runner's arrival (ETA = ceil(dist/S)),
        //    3) plant (G-1) cheap width-1 "real" gates on the other 3 sides,
        //       far enough away to be comfortably sealable in time.
        int decoySide = rnd.next(0, 3);
        int dist0 = 2 * p.S;              // ETA(decoy) = 2 exactly
        int decoyWidth = 2 * p.K + 1;      // buildTurns = 3 > ETA -> infeasible
        decoyWidth = min(decoyWidth, N - 4);

        if (decoySide == 0) { rx = max(2, min(N - 3, dist0)); ry = N / 2; }
        else if (decoySide == 1) { rx = max(2, min(N - 3, N - 1 - dist0)); ry = N / 2; }
        else if (decoySide == 2) { ry = max(2, min(N - 3, dist0)); rx = N / 2; }
        else { ry = max(2, min(N - 3, N - 1 - dist0)); rx = N / 2; }

        int mid = (decoySide == 0 || decoySide == 1) ? ry : rx;
        int start = max(1, min(N - 2 - decoyWidth, mid - decoyWidth / 2));
        placeRun(decoySide, start, decoyWidth);
        for (int pos = start; pos < start + decoyWidth; pos++) {
            int r, c;
            if (decoySide == 0) { r = 0; c = pos; }
            else if (decoySide == 1) { r = N - 1; c = pos; }
            else if (decoySide == 2) { r = pos; c = 0; }
            else { r = pos; c = N - 1; }
            allGateCells.push_back({r, c});
        }

        // the "real" gates only need to be sealable within the ~2 turns
        // before the decoy forces the escape; width 1 makes them instantly
        // sealable regardless of distance, so no extra buffer is required.
        vector<int> otherSides;
        for (int s = 0; s < 4; s++) if (s != decoySide) otherSides.push_back(s);
        for (int i = 2; i > 0; i--) { int j = rnd.next(0, i); swap(otherSides[i], otherSides[j]); }

        vector<vector<pair<int,int>>> occupied(4);
        int placed = 0, attempts = 0;
        while (placed < p.G - 1 && attempts < 400) {
            attempts++;
            int side = otherSides[placed % (int)otherSides.size()];
            int w = 1; // cheap, narrow "real" gates
            int lo = 1, hi = N - 2 - w + 1;
            if (hi < lo) continue;
            int startp = rnd.next(lo, hi);
            int end = startp + w - 1;
            bool ok = true;
            for (auto& rg : occupied[side]) if (!(end < rg.first - 1 || startp > rg.second + 1)) { ok = false; break; }
            if (!ok) continue;
            int r, c;
            if (side == 0) { r = 0; c = startp; }
            else if (side == 1) { r = N - 1; c = startp; }
            else if (side == 2) { r = startp; c = 0; }
            else { r = startp; c = N - 1; }
            occupied[side].push_back({startp, end});
            placeRun(side, startp, w);
            allGateCells.push_back({r, c});
            placed++;
        }
    } else {
        vector<vector<pair<int,int>>> occupiedRanges(4);
        vector<int> sideOrder = {0, 1, 2, 3};
        for (int i = 3; i > 0; i--) { int j = rnd.next(0, i); swap(sideOrder[i], sideOrder[j]); }
        int placed = 0, attempts = 0;
        while (placed < p.G && attempts < 500) {
            attempts++;
            int side = sideOrder[placed % 4];
            int w = rnd.next(1, 3);
            int lo = 1, hi = N - 2 - w + 1;
            if (hi < lo) continue;
            int start = rnd.next(lo, hi);
            int end = start + w - 1;
            bool ok = true;
            for (auto& rg : occupiedRanges[side]) if (!(end < rg.first - 1 || start > rg.second + 1)) { ok = false; break; }
            if (!ok) continue;
            occupiedRanges[side].push_back({start, end});
            placeRun(side, start, w);
            for (int pos = start; pos <= end; pos++) {
                int r, c;
                if (side == 0) { r = 0; c = pos; }
                else if (side == 1) { r = N - 1; c = pos; }
                else if (side == 2) { r = pos; c = 0; }
                else { r = pos; c = N - 1; }
                allGateCells.push_back({r, c});
            }
            placed++;
        }

        int cr = N / 2, cc = N / 2;
        rx = cr + rnd.next(-1, 1);
        ry = cc + rnd.next(-1, 1);
        rx = max(2, min(N - 3, rx));
        ry = max(2, min(N - 3, ry));

        auto minDistToGates = [&](int r, int c) {
            int best = INT_MAX;
            for (auto& g : allGateCells) best = min(best, max(abs(r - g.first), abs(c - g.second)));
            return best;
        };
        int need = p.S * 3 + 2;
        for (int iter = 0; iter < 8 && minDistToGates(rx, ry) < need; iter++) {
            rx = cr + (rx - cr) / 2;
            ry = cc + (ry - cc) / 2;
        }
        if (minDistToGates(rx, ry) < need) { rx = cr; ry = cc; }
    }

    rx = max(2, min(N - 3, rx));
    ry = max(2, min(N - 3, ry));
    grid[rx][ry] = '.';

    printf("%d %d %d %d %d\n", N, p.T, p.S, p.K, p.B);
    printf("%d %d\n", rx, ry);
    for (int r = 0; r < N; r++) printf("%s\n", grid[r].c_str());
    return 0;
}
