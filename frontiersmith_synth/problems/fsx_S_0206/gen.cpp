#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Covering / facility-location generator: a grid of "districts" (bakery region)
// with 4-neighbour roads plus a few long-range shortcut roads. A subset of
// districts are "shops" (demand). Each district has a build cost. The freshness
// radius r is chosen so a depot covers several nearby shops but not the whole
// region -> weighted distance-r covering is meaningful and hard.
//
// testId 1 is tiny (example scale); the grid grows to ~21x21 by testId 10.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int side = 3 + 2 * (testId - 1);          // 3, 5, 7, ..., 21
    int rows = side, cols = side;
    int n = rows * cols;

    auto node = [&](int i, int j) { return i * cols + j + 1; };

    // build costs: mostly moderate with a sprinkle of cheap and pricey districts,
    // so cost-effective covering (not just "fewest depots") matters.
    vector<int> cost(n + 1);
    for (int u = 1; u <= n; u++) {
        double p = rnd.next(0.0, 1.0);
        if (p < 0.20)      cost[u] = rnd.next(1, 3);    // cheap hubs
        else if (p < 0.85) cost[u] = rnd.next(5, 14);   // typical
        else               cost[u] = rnd.next(15, 20);  // pricey
    }

    // roads: 4-neighbour grid with travel time 1..3
    vector<array<int,3>> edges;
    auto pushEdge = [&](int u, int v) {
        int w = rnd.next(1, 3);
        edges.push_back({u, v, w});
    };
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            if (j + 1 < cols) pushEdge(node(i, j), node(i, j + 1));
            if (i + 1 < rows) pushEdge(node(i, j), node(i + 1, j));
        }

    // a few long-range shortcut roads make it a general (non-grid) graph so the
    // metric is not pure Manhattan distance -> greedy vs. local search diverge.
    int extra = side;                          // ~side shortcuts
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, rnd.next(1, 3)});
    }

    // shops (demand): ~45% of districts, at least 1
    vector<int> shops;
    double demandProb = 0.45;
    for (int u = 1; u <= n; u++)
        if (rnd.next(0.0, 1.0) < demandProb) shops.push_back(u);
    if (shops.empty()) shops.push_back(rnd.next(1, n));

    // freshness radius: with weights avg 2, a small r covers only a couple of
    // adjacent districts, so covering is tight and heuristics diverge.
    int r = 2;

    // shuffle so index order is not structural
    shuffle(edges.begin(), edges.end());
    shuffle(shops.begin(), shops.end());

    int m = (int)edges.size();
    int D = (int)shops.size();

    printf("%d %d %d %d\n", n, m, D, r);
    for (int u = 1; u <= n; u++) printf("%d%c", cost[u], u == n ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    for (int i = 0; i < D; i++) printf("%d%c", shops[i], i + 1 == D ? '\n' : ' ');
    return 0;
}
