#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Wildlife-corridor watchtower coverage generator.
// Structure ladder: testId 1 is tiny (example scale), growing to a large,
// irregular habitat grid by testId 10. Radius r = 1 keeps monitoring
// neighbourhoods small so a good cover stays a meaningful fraction of the
// build-everywhere baseline (no trivial score capping), while cost variation
// c_i in {1,2,3} makes cost-blind vs cost-aware heuristics diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int side = 3 + 2 * (testId - 1);   // 3, 5, 7, ..., 21
    int R = side, C = side;
    int N = R * C;                     // 9 .. 441

    auto node = [&](int i, int j) { return i * C + j + 1; };

    vector<pair<int,int>> edges;
    // grid trails (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i, j), node(i, j + 1)});
            if (i + 1 < R) edges.push_back({node(i, j), node(i + 1, j)});
        }

    // a few long-range corridor trails break the grid symmetry so the
    // covering structure is irregular (candidate sites cover varied cell sets)
    int extra = testId;                // 1 .. 10 extra trails
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, N), b = rnd.next(1, N);
        if (a == b) { e--; continue; }
        edges.push_back({a, b});
    }

    // per-cell installation costs in {1,2,3}; skew varies per test
    int hiP = 20 + (testId % 4) * 10;  // percent chance of a costly cell
    vector<int> cost(N + 1);
    for (int i = 1; i <= N; i++) {
        int roll = rnd.next(1, 100);
        if (roll <= hiP) cost[i] = 3;
        else if (roll <= hiP + 30) cost[i] = 2;
        else cost[i] = 1;
    }

    int r = 1;

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int M = (int)edges.size();
    printf("%d %d %d\n", N, M, r);
    for (int i = 1; i <= N; i++)
        printf("%d%c", cost[i], i == N ? '\n' : ' ');
    for (auto& e : edges)
        printf("%d %d\n", e.first, e.second);
    return 0;
}
