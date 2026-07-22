#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- Wafer Stepper Exposure Order.  testId is a difficulty/structure ladder:
// tiny sanity at 1, growing to N=2400 at 10.  Layout alternates full rectangular die
// grids, scattered wafers, a weight-needle case and a dense-cluster case.  The physical
// regime (D, alpha) alternates between "mild" (fast decay / low diffusion -- order barely
// matters, boustrophedon is fine) and "trap" (slow decay / strong diffusion -- a spatially
// contiguous scan drags a persistent hot spot across the wafer and pays badly).  Tests
// 5, 6, 7 and 10 are TRAP tests; test 4 is a weight NEEDLE; test 1 is a small full grid.

struct Field { int x, y, w; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    // ---- per-test physical regime & size ----
    // D: diffusion rate, alpha: decay rate, Q0: dose constant.
    struct Regime { int N; double D, alpha, Q0; int mode; };
    // mode: 0=full grid mild, 1=scattered mild, 2=full grid needle, 3=full grid TRAP,
    //       4=elongated TRAP, 5=cluster TRAP, 6=full grid large mild, 7=full grid large TRAP
    Regime regs[10] = {
        {   9, 1.00, 0.20,  80.0, 0},   // 1: tiny sanity, 3x3 grid
        {  40, 0.60, 0.35,  60.0, 1},   // 2: scattered, mild (fast decay)
        { 100, 0.80, 0.20, 100.0, 0},   // 3: full grid, general
        { 220, 0.90, 0.18, 100.0, 2},   // 4: full grid, weight NEEDLE
        { 420, 2.50, 0.05, 150.0, 3},   // 5: full grid, TRAP (strong diffusion, slow decay)
        { 640, 2.00, 0.05, 150.0, 4},   // 6: elongated grid, TRAP
        { 900, 1.60, 0.07, 130.0, 5},   // 7: dense cluster + sparse halo, TRAP
        {1300, 1.00, 0.15, 100.0, 0},   // 8: full grid, general at scale
        {1800, 0.90, 0.20, 100.0, 6},   // 9: large, mild
        {2400, 2.20, 0.05, 150.0, 7},   // 10: large, TRAP
    };
    Regime R = regs[idx - 1];
    int N = R.N;

    vector<Field> f;
    f.reserve(N);

    auto fullGrid = [&](int n, bool wide) {
        int rows, cols;
        if (wide) {
            rows = max(2, (int)round(sqrt((double)n / 8.0)));
            cols = (n + rows - 1) / rows;
        } else {
            cols = max(1, (int)round(sqrt((double)n)));
            rows = (n + cols - 1) / cols;
        }
        vector<pair<int,int>> cells;
        for (int r = 0; r < rows && (int)cells.size() < rows * cols; r++)
            for (int c = 0; c < cols; c++)
                cells.push_back({c, r});
        shuffle(cells.begin(), cells.end());
        cells.resize(n);
        vector<Field> out;
        out.reserve(n);
        for (auto& pr : cells) out.push_back({pr.first, pr.second, 1});
        return out;
    };

    if (R.mode == 0 || R.mode == 6) {
        f = fullGrid(N, false);
        for (auto& e : f) e.w = rnd.next(1, 5);
    } else if (R.mode == 1) {
        // scattered wafer: distinct random points in a bounded disc-ish area
        set<pair<int,int>> used;
        int C = max(20, (int)round(sqrt((double)N) * 6));
        int cx = C / 2, cy = C / 2, rad = C / 2;
        while ((int)f.size() < N) {
            int x = rnd.next(0, C), y = rnd.next(0, C);
            long long dx = x - cx, dy = y - cy;
            if (dx * dx + dy * dy > (long long)rad * rad) continue;
            if (used.insert({x, y}).second) f.push_back({x, y, rnd.next(1, 5)});
        }
    } else if (R.mode == 2) {
        // full grid + weight needle: a handful of very heavy fields amid light ones
        f = fullGrid(N, false);
        for (auto& e : f) e.w = 1;
        int needles = max(3, N / 40);
        vector<int> idxs(N);
        for (int i = 0; i < N; i++) idxs[i] = i;
        shuffle(idxs.begin(), idxs.end());
        for (int i = 0; i < needles; i++) f[idxs[i]].w = rnd.next(35, 60);
    } else if (R.mode == 3 || R.mode == 7) {
        f = fullGrid(N, false);
        for (auto& e : f) e.w = rnd.next(1, 5);
    } else if (R.mode == 4) {
        f = fullGrid(N, true);   // wide/short: long rows -> long hot trails under boustrophedon
        for (auto& e : f) e.w = rnd.next(1, 5);
    } else if (R.mode == 5) {
        // dense cluster (tight pitch, high weight) + sparse halo of far-apart light fields
        int clusterN = N * 2 / 3;
        int haloN = N - clusterN;
        int side = max(1, (int)round(sqrt((double)clusterN)));
        set<pair<int,int>> used;
        vector<pair<int,int>> cells;
        for (int r = 0; r < side + 2 && (int)cells.size() < side * side + 2 * side; r++)
            for (int c = 0; c < side; c++) cells.push_back({c, r});
        shuffle(cells.begin(), cells.end());
        cells.resize(clusterN);
        for (auto& pr : cells) { f.push_back({pr.first, pr.second, rnd.next(20, 40)}); used.insert(pr); }
        int C = side + 40;
        while ((int)f.size() < clusterN + haloN) {
            int x = rnd.next(0, C * 3), y = rnd.next(0, C * 3);
            if (x < side + 6 && y < side + 6) continue;   // stay clear of the cluster block
            if (used.insert({x, y}).second) f.push_back({x, y, rnd.next(1, 3)});
        }
    }

    // ---- shuffle field listing order so the checker's input-order baseline is arbitrary ----
    shuffle(f.begin(), f.end());

    printf("%d\n", N);
    printf("%.6f %.6f %.6f\n", R.D, R.alpha, R.Q0);
    for (auto& e : f) printf("%d %d %d\n", e.x, e.y, e.w);
    return 0;
}
