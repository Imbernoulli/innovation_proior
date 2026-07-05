#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Skyline Sprouts: rooftop garden-bed polyomino packing generator.
// testId is a difficulty ladder: tiny example-scale roof at testId 1 growing to a
// large, densely-obstructed roof at testId 10, with varying supplies and obstacle mix.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- roof size ladder ----
    int H = 10 + 6 * (testId - 1);   // 10, 16, ..., 64
    int W = 10 + 6 * (testId - 1);
    if (H > 70) H = 70;
    if (W > 70) W = 70;

    // ---- fixed catalog of planter-bed shapes (id 0 is always the 2x2 square) ----
    // each entry: list of (dr, dc), normalized so min dr = min dc = 0.
    vector<vector<pair<int,int>>> cat = {
        {{0,0},{0,1},{1,0},{1,1}},              // 0: 2x2 square (BASELINE shape)
        {{0,0},{0,1},{0,2},{0,3}},              // 1: I-tetromino
        {{0,0},{1,0},{2,0},{2,1}},              // 2: L-tetromino
        {{0,0},{0,1},{0,2},{1,1}},              // 3: T-tetromino
        {{0,1},{0,2},{1,0},{1,1}},              // 4: S-tetromino
        {{0,1},{1,0},{1,1},{1,2},{2,1}},        // 5: plus-pentomino
        {{0,0},{0,1},{0,2},{1,0},{1,1},{1,2}},  // 6: 2x3 rectangle
        {{0,0},{0,1}},                          // 7: domino
        {{0,0}}                                 // 8: monomino
    };
    int D = (int)cat.size();

    int area = H * W;

    // ---- supplies: shape 0 effectively unlimited; others bounded (knapsack tension) ----
    vector<int> supply(D);
    supply[0] = area;                    // unlimited square supply
    supply[1] = 4 + area / 60;           // I
    supply[2] = 4 + area / 55;           // L
    supply[3] = 4 + area / 55;           // T
    supply[4] = 4 + area / 55;           // S
    supply[5] = 3 + area / 90;           // plus (scarce big piece)
    supply[6] = 3 + area / 100;          // 2x3 rectangle (scarce big piece)
    supply[7] = 6 + area / 45;           // domino
    supply[8] = 3 + area / 80;           // monomino (scarce gap-filler)

    // ---- obstructed tiles: scattered, breaking clean tilings; top-left 2x2 kept free ----
    double dens = 0.08 + 0.008 * testId; // 0.088 .. 0.16
    if (dens > 0.20) dens = 0.20;
    vector<vector<char>> blocked(H, vector<char>(W, 0));
    int Q = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            // keep the top-left 2x2 region always free
            if (r <= 1 && c <= 1) continue;
            if (rnd.next(0.0, 1.0) < dens) { blocked[r][c] = 1; Q++; }
        }

    // ---- emit ----
    printf("%d %d %d\n", H, W, D);
    for (int i = 0; i < D; i++) {
        printf("%d %d\n", (int)cat[i].size(), supply[i]);
        for (auto& p : cat[i]) printf("%d %d\n", p.first, p.second);
    }
    printf("%d\n", Q);
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            if (blocked[r][c]) printf("%d %d\n", r, c);
    return 0;
}
