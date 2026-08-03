#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Multi-terminal shortest-path interdiction on a beehive foraging network.
// Structure ladder: a grid of expensive corridors overlaid with a sparse set of
// very cheap "highway" shortcuts. The cheap shortcuts carry almost all foraging
// traffic, so destroying a well-chosen few forces long expensive detours -> the
// interdiction objective is highly sensitive and heuristics diverge.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // testId 1 tiny (example scale), growing to a large grid by testId 10.
    int side = max(3, 3 * testId);          // 3, 6, 9, ..., 30
    int R = side, C = side;
    int n = R * C;                          // up to 900
    auto node = [&](int i, int j) { return i * C + j + 1; };

    vector<array<int,3>> edges;             // u, v, w  (index = position, then shuffled)

    // Expensive grid corridors (right + down neighbours).
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i,j), node(i,j+1), (int)rnd.next(25, 60)});
            if (i + 1 < R) edges.push_back({node(i,j), node(i+1,j), (int)rnd.next(25, 60)});
        }

    // Sparse cheap "highway" shortcuts (the load-bearing backbone).
    int sc = side + testId;                 // number of shortcuts (grows with scale)
    for (int e = 0; e < sc; e++) {
        int a = (int)rnd.next(1, n), b = (int)rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, (int)rnd.next(1, 4)});
    }

    int s = 1;
    int k = min(25, 1 + 2 * testId);        // interdiction budget, < number of shortcuts

    // Forage patches (terminals): scattered distinct nodes, none equal to s.
    int q = min({30, 2 * testId, n - 1});
    vector<int> pool;
    for (int v = 2; v <= n; v++) pool.push_back(v);
    shuffle(pool.begin(), pool.end());
    vector<int> term(pool.begin(), pool.begin() + q);

    // Shuffle edge order so index != structural position.
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, q, k);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    for (int i = 0; i < q; i++) printf("%d%c", term[i], i + 1 < q ? ' ' : '\n');
    return 0;
}
