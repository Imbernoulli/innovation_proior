#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // Difficulty ladder: dim grows from 4 (tiny sanity) to 40 (large).
    int dim = 4 + 4 * (t - 1);
    if (dim > 40) dim = 40;
    int R = dim, C = dim;
    int n = R * C;

    int Wlo = 8, Whi = 25;          // expensive-edge weight range
    auto id = [&](int r, int c) { return r * C + c + 1; };

    // Each edge: (u, v, w, cost)
    vector<array<long long, 4>> edges;

    // Horizontal edges; top row (r==0) is the cheap "fast current" trunk (w=1).
    for (int r = 0; r < R; r++)
        for (int c = 0; c + 1 < C; c++) {
            long long w = (r == 0) ? 1 : rnd.next(Wlo, Whi);
            long long cost = rnd.next(1, 8);
            edges.push_back({(long long)id(r, c), (long long)id(r, c + 1), w, cost});
        }
    // Vertical edges; last column (c==C-1) is the cheap trunk (w=1).
    for (int r = 0; r + 1 < R; r++)
        for (int c = 0; c < C; c++) {
            long long w = (c == C - 1) ? 1 : rnd.next(Wlo, Whi);
            long long cost = rnd.next(1, 8);
            edges.push_back({(long long)id(r, c), (long long)id(r + 1, c), w, cost});
        }
    // Extra random survey chords (all expensive), to enrich reroute structure.
    int extra = n / 2;
    for (int i = 0; i < extra; i++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) continue;
        long long w = rnd.next(Wlo, Whi), cost = rnd.next(1, 8);
        edges.push_back({(long long)a, (long long)b, w, cost});
    }

    // Shuffle edge order so trunk edges are not trivially identifiable by index.
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();

    // Budget: ~30% of total trunk (w==1) closure cost -> forces selection.
    long long totalTrunkCost = 0;
    for (auto& e : edges)
        if (e[2] == 1) totalTrunkCost += e[3];
    long long budget = max(2LL, (long long)(0.30 * (double)totalTrunkCost));

    int s = 1, tsink = n;

    printf("%d %d %d %d %lld\n", n, m, s, tsink, budget);
    for (auto& e : edges)
        printf("%lld %lld %lld %lld\n", e[0], e[1], e[2], e[3]);

    return 0;
}
