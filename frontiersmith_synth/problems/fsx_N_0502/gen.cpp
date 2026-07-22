#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

struct Basin {
    int r, c, d;
};

struct Pump {
    int r, c, cost, reach;
};

static int clampInt(int x, int lo, int hi) {
    return max(lo, min(hi, x));
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int Hs[10] = {8, 12, 16, 20, 26, 32, 38, 46, 52, 58};
    const int Ws[10] = {10, 14, 18, 24, 30, 36, 44, 52, 58, 60};
    const int Ns[10] = {12, 28, 45, 70, 105, 145, 190, 250, 320, 380};
    int H = Hs[testId - 1];
    int W = Ws[testId - 1];
    int N = Ns[testId - 1];
    int mode = testId % 5; // 0 mixed, 1 TRAP, 2 NEEDLE, 3 PLANTED, 4 skewed trap

    vector<string> grid(H, string(W, '6'));
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int w = rnd.next(5, 8);
            grid[r][c] = char('0' + w);
        }
    }

    // Relief canals on borders and district seams.
    for (int c = 0; c < W; c++) grid[0][c] = grid[H - 1][c] = '9';
    for (int r = 0; r < H; r++) grid[r][0] = grid[r][W - 1] = '9';
    int r1 = H / 4, r2 = (3 * H) / 4, c1 = W / 4, c2 = (3 * W) / 4;
    for (int c = 0; c < W; c++) grid[r1][c] = grid[r2][c] = '9';
    for (int r = 0; r < H; r++) grid[r][c1] = grid[r][c2] = '9';

    // Low-width trap seams: shortest routes into central pumps cross these, but relief
    // canals around the seams are much wider.
    int midR = H / 2, midC = W / 2;
    for (int r = 1; r + 1 < H; r++) {
        grid[r][midC] = '1';
        if (mode == 1 || mode == 4) grid[r][max(1, midC - 1)] = '2';
    }
    for (int c = 1; c + 1 < W; c++) {
        grid[midR][c] = '1';
        if (mode == 2 || mode == 4) grid[max(1, midR - 1)][c] = '2';
    }
    grid[midR][midC] = '1';

    int regRows = (testId <= 3 ? 2 : (testId <= 7 ? 3 : 4));
    int regCols = (testId <= 4 ? 2 : (testId <= 8 ? 3 : 4));
    vector<pair<int,int>> centers;
    for (int rr = 0; rr < regRows; rr++) {
        for (int cc = 0; cc < regCols; cc++) {
            int r = (2 * rr + 1) * H / (2 * regRows);
            int c = (2 * cc + 1) * W / (2 * regCols);
            centers.push_back({clampInt(r, 1, H - 2), clampInt(c, 1, W - 2)});
        }
    }

    vector<Basin> basins;
    basins.reserve(N);
    set<pair<int,int>> used;
    int jitterR = max(2, H / (2 * regRows) - 1);
    int jitterC = max(2, W / (2 * regCols) - 1);
    for (int i = 0; i < N; i++) {
        auto ctr = centers[i % centers.size()];
        int r = ctr.first, c = ctr.second;
        bool placed = false;
        for (int tries = 0; tries < 200 && !placed; tries++) {
            int nr = clampInt(ctr.first + rnd.next(-jitterR, jitterR), 1, H);
            int nc = clampInt(ctr.second + rnd.next(-jitterC, jitterC), 1, W);
            if (!used.count({nr, nc})) {
                r = nr;
                c = nc;
                placed = true;
            }
        }
        if (!placed) {
            for (int tries = 0; tries < 500 && !placed; tries++) {
                int nr = rnd.next(1, H);
                int nc = rnd.next(1, W);
                if (!used.count({nr, nc})) {
                    r = nr;
                    c = nc;
                    placed = true;
                }
            }
        }
        used.insert({r, c});
        int d = rnd.next(3, 10);
        if (mode == 2 && i < max(4, N / 8)) d = rnd.next(15, 24); // NEEDLE cluster
        if ((mode == 1 || mode == 4) && abs(c - midC) <= W / 5) d += rnd.next(3, 8);
        basins.push_back({r, c, d});
    }

    vector<Pump> pumps;
    pumps.reserve(N + 180);
    for (const auto& b : basins) {
        int cost = 4600 + rnd.next(0, 2800) + 70 * b.d;
        pumps.push_back({b.r, b.c, cost, 0});
    }

    auto addPump = [&](int r, int c, int cost, int reach) {
        r = clampInt(r, 1, H);
        c = clampInt(c, 1, W);
        cost = clampInt(cost, 1, 10000);
        reach = clampInt(reach, 0, H + W);
        pumps.push_back({r, c, cost, reach});
    };

    // TRAP pumps: very cheap and high radius, but centered on low-width canals.
    addPump(midR + 1, midC + 1, rnd.next(120, 260), H + W);
    addPump(max(1, midR), midC + 1, rnd.next(150, 320), H + W);
    if (testId >= 4) addPump(midR + 1, max(1, midC), rnd.next(180, 360), H + W);
    if (mode == 1 || mode == 4) {
        addPump(midR + 1, clampInt(midC - 2, 1, W), rnd.next(110, 240), H + W);
        addPump(clampInt(midR - 2, 1, H), midC + 1, rnd.next(110, 240), H + W);
    }

    // PLANTED district pumps: cheap enough that a spread solution is good, but each covers
    // mostly its own basin cluster.
    int districtReach = max(5, max(H / regRows, W / regCols) / 2 + 4);
    for (size_t idx = 0; idx < centers.size(); idx++) {
        int r = centers[idx].first;
        int c = centers[idx].second;
        int base = (mode == 3 ? rnd.next(230, 460) : rnd.next(420, 880));
        addPump(r, c, base, districtReach);
        if (testId >= 5) {
            addPump(clampInt(r + rnd.next(-2, 2), 1, H),
                    clampInt(c + rnd.next(-2, 2), 1, W),
                    base + rnd.next(40, 280), max(4, districtReach - rnd.next(1, 3)));
        }
    }

    // NEEDLE relief pumps near the widest perimeter corridors for high-demand clusters.
    if (mode == 2 || testId >= 6) {
        addPump(1, max(2, W / 3), rnd.next(260, 520), H + W / 3);
        addPump(H, min(W - 1, 2 * W / 3), rnd.next(260, 520), H + W / 3);
        addPump(max(2, H / 3), 1, rnd.next(260, 520), W + H / 3);
        addPump(min(H - 1, 2 * H / 3), W, rnd.next(260, 520), W + H / 3);
    }

    // Distractors and extra useful pumps fill out the larger instances.
    int extra = 5 + testId * 6;
    for (int k = 0; k < extra; k++) {
        int r = rnd.next(1, H);
        int c = rnd.next(1, W);
        int reach = rnd.next(max(3, districtReach / 2), min(H + W, districtReach * 2 + 6));
        int cost;
        if (k % 5 == 0) cost = rnd.next(900, 1900);      // broad but not as cheap as traps
        else cost = rnd.next(450, 1300);
        if (mode == 0 && k % 7 == 0) reach = H + W;
        addPump(r, c, cost, reach);
    }

    int P = (int)pumps.size();
    if (P > 650) {
        pumps.resize(650);
        P = 650;
    }

    printf("%d %d %d %d\n", H, W, N, P);
    for (int r = 0; r < H; r++) printf("%s\n", grid[r].c_str());
    for (const auto& b : basins) printf("%d %d %d\n", b.r, b.c, b.d);
    for (const auto& p : pumps) printf("%d %d %d %d\n", p.r, p.c, p.cost, p.reach);
    return 0;
}
