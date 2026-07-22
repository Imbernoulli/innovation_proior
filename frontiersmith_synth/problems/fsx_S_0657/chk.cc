#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// chk.cc -- scorer for Toroidal Sidon Beacon Placement.
// Feasibility: marks are distinct cells of an H x W torus such that every ORDERED pairwise
// difference vector ((r_a-r_b) mod H, (c_a-c_b) mod W) is distinct (the planar/torus Sidon,
// a.k.a. B2, condition).
// Objective (maximize): F = S * (1 + LAMBDA * coverage), where S is the sum of weights of the
// marked cells and coverage = m*(m-1)/(N-1) in [0,1] is the fraction of the N-1 nonzero
// difference vectors realized (coverage = 1 exactly iff the mark set is a perfect/planar
// difference set -- the Singer-construction ceiling).
// Baseline B (trivial, max): the single heaviest cell in the grid (m=1, coverage=0, F=that
// weight) -- always feasible and positive.

int main(int argc, char *argv[]) {
    registerTestlibCmd(argc, argv);

    int H = inf.readInt();
    int W = inf.readInt();
    long long LNUM = inf.readLong();
    long long LDEN = inf.readLong();
    if (LDEN <= 0) quitf(_fail, "bad input: LDEN<=0");
    double LAMBDA = (double)LNUM / (double)LDEN;

    long long N = (long long)H * (long long)W;
    vector<vector<long long>> wgt(H, vector<long long>(W));
    long long B = 0;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            long long v = inf.readLong();
            wgt[i][j] = v;
            if (v > B) B = v;
        }
    }
    if (B <= 0) B = 1;

    // Bounded cap on the number of marks the participant may claim: any Sidon set on a group
    // of order N obeys m*(m-1) <= N-1, so cap generously above that theoretical ceiling; this
    // keeps the O(m^2) feasibility check linear in the input size even for adversarial output.
    double mmaxD = (1.0 + sqrt(max(0.0, 1.0 + 4.0 * (double)(N - 1)))) / 2.0;
    long long MCAP = (long long)floor(mmaxD) + 5;
    if (MCAP > N) MCAP = N;
    if (MCAP < 1) MCAP = 1;

    int m = ouf.readInt(0, (int)MCAP, "m");
    vector<pair<int, int>> marks(m);
    set<pair<int, int>> seen;
    for (int t = 0; t < m; t++) {
        int r = ouf.readInt(0, H - 1, "r");
        int c = ouf.readInt(0, W - 1, "c");
        if (!seen.insert({r, c}).second) quitf(_wa, "duplicate mark at (%d,%d)", r, c);
        marks[t] = {r, c};
    }
    if (!ouf.seekEof()) quitf(_wa, "trailing tokens after the mark list");

    {
        set<long long> usedDiff;
        for (int a = 0; a < m; a++) {
            for (int b = 0; b < m; b++) {
                if (a == b) continue;
                int dr = ((marks[a].first - marks[b].first) % H + H) % H;
                int dc = ((marks[a].second - marks[b].second) % W + W) % W;
                long long id = (long long)dr * (long long)W + dc;
                if (!usedDiff.insert(id).second)
                    quitf(_wa, "two mark pairs share difference vector (%d,%d) mod (%d,%d)", dr, dc, H, W);
            }
        }
    }

    long long S = 0;
    for (auto &p : marks) S += wgt[p.first][p.second];

    double coverage = (N > 1) ? ((double)m * (double)(m - 1)) / (double)(N - 1) : 0.0;
    double F = (double)S * (1.0 + LAMBDA * coverage);

    double sc = min(1000.0, 100.0 * F / max(1e-9, (double)B));
    quitp(sc / 1000.0, "OK m=%d S=%lld B=%lld coverage=%.4f F=%.3f Ratio: %.6f", m, S, B, coverage, F, sc / 1000.0);
    return 0;
}
