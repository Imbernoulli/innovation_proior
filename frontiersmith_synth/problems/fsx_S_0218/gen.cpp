#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: rectangular greenhouse grid of climate zones ----
    // testId 1 is tiny (example scale); grows to a large greenhouse by testId 10.
    int R = 3 + 3 * (testId - 1);   // 3, 6, 9, ..., 30
    int C = 3 + 4 * (testId - 1);   // 3, 7, 11, ..., 39
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // radius alternates between 1 and 2 so coverage structure varies across tests
    int r = (testId % 2 == 1) ? 1 : 2;

    vector<pair<int,int>> edges;

    // grid ducts (right + down neighbours)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i, j), node(i, j + 1)});
            if (i + 1 < R) edges.push_back({node(i, j), node(i + 1, j)});
        }

    // extra long-range ducts (shortcuts) create irregular high-coverage "hub" zones
    int extra = 3 * testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b});
    }

    // heavy-tailed installation costs: most zones cheap, a minority expensive
    vector<int> cost(n + 1);
    for (int v = 1; v <= n; v++) {
        if (rnd.next(0.0, 1.0) < 0.15)
            cost[v] = rnd.next(40, 100);
        else
            cost[v] = rnd.next(1, 20);
    }

    // shuffle duct order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, r);
    for (int v = 1; v <= n; v++)
        printf("%d%c", cost[v], v == n ? '\n' : ' ');
    for (auto& e : edges)
        printf("%d %d\n", e.first, e.second);
    return 0;
}
