#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Aim One Ray to Sweep a Mirrored Grid"
// Ladder: R,C grow from small sanity cases up to 150x150. Several tests are
// engineered so that R-1,C-1 share a large gcd, which locks the mirror-free
// 45-degree billiard ray from S onto a small sub-orbit (unfolds to a straight
// line whose lattice period is governed by gcd(R-1,C-1)) long before it can
// cover the board. High-value cells are then planted OUTSIDE that default
// orbit, so a purely reactive/local mirror policy (which never reasons about
// the orbit's closing state) stays stuck near the small orbit, while a
// solver that understands the billiard-unfold/gcd structure can splice a
// mirror in at the right point to reach the planted value.
//
// S is always placed on a boundary EDGE cell that is NOT a corner, so the
// solver's initial-direction choice (UL/UR/DL/DR) is genuinely 2-way
// meaningful (a corner start collapses all 4 choices to the same direction
// after the first wall bounce).

static int R, C;
static vector<string> grid;

// Same billiard simulation as the checker's baseline (DIR=DR, zero mirrors),
// used here only to find which cells the naive default orbit reaches, so we
// know where to plant the high-value trap material.
static vector<char> reachableSet(int sr, int sc) {
    vector<char> visitedCell(R * C, 0);
    vector<char> visitedState(R * C * 4, 0);
    int dr = 1, dc = 1, r = sr, c = sc;
    auto enc = [&](int a, int b) { return ((a == 1) ? 1 : 0) * 2 + ((b == 1) ? 1 : 0); };
    visitedCell[r * C + c] = 1;
    visitedState[(r * C + c) * 4 + enc(dr, dc)] = 1;
    long long maxSteps = 4LL * R * C + 10;
    for (long long step = 0; step < maxSteps; step++) {
        if (r + dr < 0 || r + dr >= R) dr = -dr;
        if (c + dc < 0 || c + dc >= C) dc = -dc;
        int nr = r + dr, nc = c + dc;
        if (nr < 0 || nr >= R || nc < 0 || nc >= C) break;
        if (grid[nr][nc] == '#') break;
        r = nr; c = nc;
        int stIdx = (r * C + c) * 4 + enc(dr, dc);
        if (visitedState[stIdx]) break;
        visitedState[stIdx] = 1;
        if (!visitedCell[r * C + c]) visitedCell[r * C + c] = 1;
    }
    return visitedCell;
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int Bbudget;
    string trapMode; // "none" | "cluster" | "needle"
    switch (testId) {
        case 1: R = 9;   C = 7;   Bbudget = 2;  trapMode = "none";    break;
        case 2: R = 15;  C = 11;  Bbudget = 3;  trapMode = "none";    break;
        case 3: R = 23;  C = 17;  Bbudget = 4;  trapMode = "none";    break;
        case 4: R = 31;  C = 21;  Bbudget = 5;  trapMode = "needle";  break; // gcd(30,20)=10
        case 5: R = 41;  C = 31;  Bbudget = 6;  trapMode = "cluster"; break; // gcd(40,30)=10
        case 6: R = 51;  C = 41;  Bbudget = 7;  trapMode = "needle";  break; // gcd(50,40)=10
        case 7: R = 56;  C = 45;  Bbudget = 8;  trapMode = "cluster"; break; // gcd(55,44)=11
        case 8: R = 78;  C = 56;  Bbudget = 10; trapMode = "needle";  break; // gcd(77,55)=11
        case 9: R = 101; C = 71;  Bbudget = 12; trapMode = "cluster"; break; // gcd(100,70)=10
        default: R = 136; C = 100; Bbudget = 14; trapMode = "needle"; break; // gcd(135,99)=9
    }

    // ---- boundary source location. Trap tests (4..10) place S at a CORNER:
    // the mirror-free billiard orbit is most tightly locked onto a small
    // gcd(R-1,C-1)-governed sub-lattice exactly when S starts at a corner
    // (an off-corner edge start has a "generic" unfolding phase and, we
    // verified empirically, ends up covering most of the reachable board
    // regardless of gcd -- it is NOT a trap). Non-trap sanity tests (1..3)
    // use a random non-corner edge cell so the DIR choice is genuinely
    // 2-way meaningful there. ----
    int sr, sc;
    if (trapMode != "none") {
        int corner = testId % 4;
        sr = (corner < 2) ? 0 : R - 1;
        sc = (corner % 2 == 0) ? 0 : C - 1;
    } else {
        int side = testId % 4;
        if (side == 0) { sr = 0; sc = 1 + rnd.next(0, C - 3); }
        else if (side == 1) { sr = R - 1; sc = 1 + rnd.next(0, C - 3); }
        else if (side == 2) { sr = 1 + rnd.next(0, R - 3); sc = 0; }
        else { sr = 1 + rnd.next(0, R - 3); sc = C - 1; }
    }

    // ---- block density: zero on the engineered trap tests (blocks would
    // otherwise randomly kill the baseline orbit at step 0 near a boundary
    // S, swamping the intended gcd(R-1,C-1)-period trap with pure luck);
    // a little light seasoning is kept on the small non-trap sanity tests,
    // guarded away from S so the baseline orbit still gets going ----
    double blockProb = (trapMode == "none") ? 0.02 : 0.0;
    grid.assign(R, string(C, '.'));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (i == sr && j == sc) continue;
            if (max(abs(i - sr), abs(j - sc)) <= 4) continue; // keep S's neighborhood clear
            if (rnd.next(0.0, 1.0) < blockProb) grid[i][j] = '#';
        }
    grid[sr][sc] = 'S';

    // ---- weights: baseline small values, everywhere except blocked ----
    vector<vector<int>> W(R, vector<int>(C, 0));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++)
            if (grid[i][j] != '#') W[i][j] = rnd.next(2, 6);

    if (trapMode != "none") {
        vector<char> reach = reachableSet(sr, sc);
        vector<pair<int,int>> cand;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++)
                if (grid[i][j] != '#' && !reach[i * C + j]) cand.push_back({i, j});

        // Keep the trap's TOTAL extra value modest relative to the (small)
        // default-orbit baseline, so a solver that fully unlocks it lands in
        // a healthy 0.5-0.9 ratio band instead of slamming the 10x score cap.
        if (!cand.empty()) {
            if (trapMode == "needle") {
                int cnt = min((int)cand.size(), max(3, (R * C) / 170));
                for (int t = (int)cand.size() - 1; t > 0; t--) {
                    int j = rnd.next(0, t);
                    swap(cand[t], cand[j]);
                }
                for (int t = 0; t < cnt; t++) {
                    auto [i, j] = cand[t];
                    W[i][j] = rnd.next(9, 14);
                }
            } else { // cluster
                int anchorIdx = rnd.next(0, (int)cand.size() - 1);
                auto [ai, aj] = cand[anchorIdx];
                int radius = max(3, min(R, C) / 6);
                int boosted = 0, cap = max(8, (R * C) / 60);
                for (int i = max(0, ai - radius); i <= min(R - 1, ai + radius) && boosted < cap; i++)
                    for (int j = max(0, aj - radius); j <= min(C - 1, aj + radius) && boosted < cap; j++)
                        if (grid[i][j] != '#' && !reach[i * C + j]) {
                            W[i][j] = rnd.next(7, 12);
                            boosted++;
                        }
            }
        }
    }

    // ---- emit ----
    printf("%d %d %d\n", R, C, Bbudget);
    printf("%d %d\n", sr, sc);
    for (int i = 0; i < R; i++) printf("%s\n", grid[i].c_str());
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) printf("%d%c", W[i][j], j + 1 == C ? '\n' : ' ');
    }
    return 0;
}
