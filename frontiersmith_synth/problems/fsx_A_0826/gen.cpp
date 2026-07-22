#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Biogas Villages: Digesters versus Pipelines".
// testId is a difficulty/structure ladder (tiny at 1 -> M~1950 at 10).
//
// STRUCTURE: farms are planted into G "groups". Each group g has V villages arranged on a
// small ring of radius gap(g) around the group's centre; each village has F farms scattered
// within a much smaller radius (farmSpread) around its own village centre. Group centres are
// placed far apart from every other group (never a merge candidate across groups).
//
// THE TRAP: gap(g) alternates by group parity -- even g is TIGHT, odd g is LOOSE -- but the
// GLOBAL cost parameters A, Bc, Lm (fixed cost / concave coefficient / pipeline loss rate)
// are IDENTICAL for every group in the instance. gap(g) is derived from a closed-form
// concave-vs-linear crossover estimate using those same A, Bc, Lm, so TIGHT groups sit
// comfortably below the crossover (regional consolidation genuinely saves money) and LOOSE
// groups sit comfortably above it (consolidation genuinely loses money) -- in the SAME test
// file. A solver using one fixed clustering radius for the whole map cannot get both right;
// only reading each group's own spacing against the instance's own A/Bc/Lm gets it right.
// testId 8 also plants a NEEDLE: one group far tighter than the rest, standing out sharply.
//
// Deterministic: all randomness comes from testlib `rnd`, seeded by testId.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    int Garr[10]  = {2,    2,    3,    3,    4,    5,    6,    8,    10,   15};
    int Varr[10]  = {2,    2,    2,    3,    3,    3,    3,    4,    4,    5};
    int Farr[10]  = {2,    6,    9,    12,   14,   16,   18,   20,   22,   26};
    long long Aarr[10]  = {4000, 4000, 4500, 5000, 5000, 5500, 5500, 6000, 6000, 6000};
    long long Bcarr[10] = {150,  150,  180,  200,  200,  220,  220,  250,  250,  250};
    long long Lmarr[10] = {12,   12,   11,   10,   10,   9,    9,    8,    8,    8};

    int G = Garr[idx - 1], V = Varr[idx - 1], F = Farr[idx - 1];
    long long A = Aarr[idx - 1], Bc = Bcarr[idx - 1], Lm = Lmarr[idx - 1];
    double L = Lm / 1000.0;

    const double BOARD = 100000.0;
    const double avgW = 100.0;
    double capV = F * avgW;                 // approx capacity of one village
    double Ctot = V * capV;                 // approx capacity of one fully-merged group

    // closed-form crossover estimate: merging V equal-capacity villages of a group saves
    //   dFacility = (V-1)*A + Bc*Ctot^0.6*(V^0.4 - 1)     [concave facility-cost saving]
    // and costs approximately
    //   dTransport(gap) = L*Ctot*(1-1/V)*gap                [extra pipeline loss]
    // so the break-even inter-village spacing is thresholdGap = dFacility / (L*Ctot*(1-1/V)).
    double dFacility = (V - 1) * (double)A + (double)Bc * pow(Ctot, 0.6) * (pow((double)V, 0.4) - 1.0);
    double denom = L * Ctot * (1.0 - 1.0 / V);
    double thresholdGap = (denom > 1e-9) ? (dFacility / denom) : 5000.0;
    thresholdGap = max(300.0, min(thresholdGap, 30000.0));

    double tightGap = max(60.0, thresholdGap * 0.35);
    double looseGap = thresholdGap * 2.6 + 500.0;

    // farm spread inside a village must stay well below tightGap (village compactness)
    double farmSpread = max(15.0, min(tightGap * 0.10, 220.0));

    // lay group centres out on a grid, well separated from each other
    int cols = (int)ceil(sqrt((double)G));
    int rows = (G + cols - 1) / cols;
    double margin = 7000.0;
    double cellW = (BOARD - 2 * margin) / max(1, cols);
    double cellH = (BOARD - 2 * margin) / max(1, rows);
    double cellSize = min(cellW, cellH);

    // clamp gaps so a group's ring can never reach a neighbouring group's cell
    double maxAllowedGap = cellSize / (2.6 * max(1, V));
    tightGap = min(tightGap, maxAllowedGap * 0.45);
    looseGap = min(looseGap, maxAllowedGap * 0.92);
    if (looseGap <= tightGap * 1.6) looseGap = tightGap * 1.6 + 200.0;
    farmSpread = min(farmSpread, tightGap * 0.12);
    farmSpread = max(farmSpread, 8.0);

    vector<long long> X, Y, W;
    X.reserve(G * V * F + 8);
    Y.reserve(G * V * F + 8);
    W.reserve(G * V * F + 8);

    for (int g = 0; g < G; g++) {
        int r = g / cols, c = g % cols;
        double gx = margin + cellW * (c + 0.5);
        double gy = margin + cellH * (r + 0.5);
        // small per-group jitter of the centre, well inside the cell
        gx += rnd.next(-cellSize * 0.15, cellSize * 0.15);
        gy += rnd.next(-cellSize * 0.15, cellSize * 0.15);

        bool tight = (g % 2 == 0);
        double gap = tight ? tightGap : looseGap;
        // testId 8 plants an explicit NEEDLE: group 0 is far tighter than every other
        // (still-alternating) group, so it stands out sharply amid mostly-loose neighbours.
        if (idx == 8 && g == 0) gap = max(40.0, tightGap * 0.22);

        // V villages sit on a ring; NEAREST-neighbour spacing on that ring is
        // 2*ringR*sin(pi/V), so solve ringR so the nearest-village spacing equals `gap`
        // (the threshold formula assumed spacing ~= gap directly).
        double ringR = gap / (2.0 * sin(acos(-1.0) / max(2, V)));

        double ang0 = rnd.next(0.0, 2.0 * acos(-1.0));
        for (int v = 0; v < V; v++) {
            double ang = ang0 + 2.0 * acos(-1.0) * v / V;
            double vx = gx + ringR * cos(ang);
            double vy = gy + ringR * sin(ang);
            for (int f = 0; f < F; f++) {
                double fx = vx + rnd.next(-farmSpread, farmSpread);
                double fy = vy + rnd.next(-farmSpread, farmSpread);
                long long ix = (long long)llround(fx);
                long long iy = (long long)llround(fy);
                ix = min((long long)BOARD, max(0LL, ix));
                iy = min((long long)BOARD, max(0LL, iy));
                long long w = rnd.next(50, 150);
                X.push_back(ix);
                Y.push_back(iy);
                W.push_back(w);
            }
        }
    }

    int M = (int)X.size();
    vector<int> perm(M);
    for (int i = 0; i < M; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());

    printf("%d %lld %lld %lld\n", M, A, Bc, Lm);
    for (int t = 0; t < M; t++) {
        int i = perm[t];
        printf("%lld %lld %lld\n", X[i], Y[i], W[i]);
    }
    return 0;
}
