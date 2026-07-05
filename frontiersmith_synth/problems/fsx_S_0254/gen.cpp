#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Tide-pool covering-facility-location generator.
// A rocky shore is a grid of tide pools connected by channels (weighted edges),
// plus a few long tidal channels (shortcuts). We place sensors that monitor every
// pool within total channel-distance R; the participant must cover all pools at
// minimum total sensor cost. testId is a difficulty/structure ladder.
int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // grid side grows from tiny (example scale) to large.
    int sides[11] = {0, 3, 12, 22, 32, 44, 56, 68, 78, 86, 92};
    int side = sides[min(testId, 10)];
    int Rr = side, Cc = side;
    int N = Rr * Cc;

    auto node = [&](int i, int j) { return i * Cc + j + 1; };

    // covering radius (in channel distance). Kept small so a sensor covers only a
    // local cluster of pools -> covering needs a substantial, spread-out set and
    // the everywhere-baseline / achieved-cost ratio stays well under the 10x cap.
    int R = 2 + (testId % 2);               // 2 or 3

    auto chanW = [&]() -> int { return rnd.next(1, 2); }; // short hops

    vector<array<int,3>> edges; // u v w
    for (int i = 0; i < Rr; i++)
        for (int j = 0; j < Cc; j++) {
            if (j + 1 < Cc) edges.push_back({node(i,j), node(i,j+1), chanW()});
            if (i + 1 < Rr) edges.push_back({node(i,j), node(i+1,j), chanW()});
        }

    // a few long tidal channels (random shortcuts) add alternate cover geometry.
    int extra = 2 * testId;
    for (int e = 0; e < extra && N >= 2; e++) {
        int a = rnd.next(1, N), b = rnd.next(1, N);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, rnd.next(2, 5)});
    }

    // sensor installation costs: mildly skewed but bounded, so cost matters yet the
    // objective cannot collapse (weighted covering, not pure unit set cover).
    vector<int> cost(N + 1);
    for (int v = 1; v <= N; v++) {
        if (rnd.next(0.0, 1.0) < 0.15) cost[v] = rnd.next(9, 14); // premium sites
        else cost[v] = rnd.next(2, 8);                           // ordinary sites
    }

    shuffle(edges.begin(), edges.end());
    int M = (int)edges.size();

    printf("%d %d %d\n", N, M, R);
    for (int v = 1; v <= N; v++) printf("%d%c", cost[v], v == N ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
