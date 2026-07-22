#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- Single-Crystal Blade Casting scorer.
//
// Simulates competitive anisotropic growth fronts from the contestant's seeds via a
// multi-source Dijkstra over the 8-neighbour grid (deterministic tie-break: lower seed
// index wins simultaneous arrival).  Objective (minimize):
//     F = G + LAMBDA * M
// G = grain-boundary length = # of orthogonally-adjacent in-region cell pairs whose
//     nearest seed differs.
// M = floor(max arrival time over all region cells / 1000)  ("peak solidification time").
// Baseline B = F of the single-seed-at-first-region-cell / orientation-0 construction
// (always feasible since K >= 1).  ratio = min(1, (B / max(1,F)) / 10).

static int H, W, K;
static long long LAMBDA;
static vector<string> grid; // '#' or '0'..'4'

static inline bool inRegion(int r, int c) {
    return r >= 0 && r < H && c >= 0 && c < W && grid[r][c] != '#';
}

// 8 directions with fixed angles (degrees), row increases "south".
static int DR[8] = {0, -1, -1, -1, 0, 1, 1, 1};
static int DC[8] = {1, 1, 0, -1, -1, -1, 0, 1};
static int ANG[8] = {0, 45, 90, 135, 180, 225, 270, 315};
static long long BASELEN[8] = {1000, 1414, 1000, 1414, 1000, 1414, 1000, 1414};

static inline int factorFor(int dirIdx, int phi) {
    int dirAngle = ANG[dirIdx];
    int phiAngle = 15 * phi;
    int diff = ((dirAngle - phiAngle) % 90 + 90) % 90;
    int delta = min(diff, 90 - diff); // in {0,15,30,45}
    return 100 + (150 * delta) / 45;
}

static inline int zoneR(int r, int c) {
    int d = grid[r][c] - '0';
    return 100 + 50 * d;
}

// Runs the competitive growth simulation for the given seeds (r,c,phi), each seed's index
// in this vector is its priority in ties. Returns F = G + LAMBDA*M.
static long long simulate(const vector<array<int,3>>& seeds) {
    vector<long long> dist(H * (long long)W, LLONG_MAX);
    vector<int> owner(H * (long long)W, -1);
    vector<char> visited(H * (long long)W, 0);
    // min-heap of (dist, seedIdx, r, c)
    typedef tuple<long long, int, int, int> T;
    priority_queue<T, vector<T>, greater<T>> pq;
    for (int i = 0; i < (int)seeds.size(); i++) {
        int r = seeds[i][0], c = seeds[i][1];
        pq.push({0LL, i, r, c});
    }
    long long maxDist = 0;
    while (!pq.empty()) {
        auto [d, idx, r, c] = pq.top(); pq.pop();
        int cell = r * W + c;
        if (visited[cell]) continue;
        visited[cell] = 1;
        dist[cell] = d;
        owner[cell] = idx;
        maxDist = max(maxDist, d);
        int phi = seeds[idx][2];
        for (int k = 0; k < 8; k++) {
            int nr = r + DR[k], nc = c + DC[k];
            if (!inRegion(nr, nc)) continue;
            int ncell = nr * W + nc;
            if (visited[ncell]) continue;
            int fac = factorFor(k, phi);
            long long R = zoneR(nr, nc);
            long long cost = (BASELEN[k] * R * (long long)fac) / 10000;
            if (cost < 1) cost = 1;
            long long nd = d + cost;
            pq.push({nd, idx, nr, nc});
        }
    }
    // grain-boundary length: 4-neighbour (orthogonal) pairs with different owner
    long long G = 0;
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++) {
            if (!inRegion(r, c)) continue;
            int cell = r * W + c;
            // right neighbour
            if (inRegion(r, c + 1)) {
                int oc = r * W + (c + 1);
                if (owner[cell] != owner[oc]) G++;
            }
            // down neighbour
            if (inRegion(r + 1, c)) {
                int oc = (r + 1) * W + c;
                if (owner[cell] != owner[oc]) G++;
            }
        }
    long long M = maxDist / 1000;
    long long F = G + LAMBDA * M;
    return F;
}

int main(int argc, char* argv[]) {
    registerTestlibCmd(argc, argv);

    H = inf.readInt();
    W = inf.readInt();
    K = inf.readInt();
    LAMBDA = inf.readLong();
    grid.assign(H, "");
    for (int r = 0; r < H; r++) grid[r] = inf.readToken();
    for (int r = 0; r < H; r++)
        if ((int)grid[r].size() != W) quitf(_fail, "bad test data: row %d wrong width", r);

    int firstR = -1, firstC = -1;
    for (int r = 0; r < H && firstR < 0; r++)
        for (int c = 0; c < W; c++)
            if (grid[r][c] != '#') { firstR = r; firstC = c; break; }
    if (firstR < 0) quitf(_fail, "bad test data: empty region");

    // ---- read participant's seeds ----
    int m = ouf.readInt(1, K, "m");
    vector<array<int,3>> seeds;
    set<pair<int,int>> used;
    for (int i = 0; i < m; i++) {
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        int p = ouf.readInt(0, 5, "phi");
        if (!inRegion(r, c)) quitf(_wa, "seed %d at (%d,%d) is outside the casting region", i, r, c);
        if (!used.insert({r, c}).second) quitf(_wa, "duplicate seed cell (%d,%d)", r, c);
        seeds.push_back({r, c, p});
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing output after the seed list");

    long long F = simulate(seeds);

    vector<array<int,3>> baseSeeds = {{firstR, firstC, 0}};
    long long B = simulate(baseSeeds);
    if (B <= 0) B = 1;

    double sc = min(1000.0, 100.0 * (double)B / (double)max(1LL, F));
    quitp(sc / 1000.0, "OK F=%lld B=%lld Ratio: %.6f", F, B, sc / 1000.0);
    return 0;
}
