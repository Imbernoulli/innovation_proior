#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Alpine Lift Network Coverage -- generator.
// The mountain is a GEOMETRIC mesh: zones sit on a jittered ~sqrt(N) x sqrt(N) grid
// (like slope junctions), slopes connect neighbouring cells with ski-time ~ their
// spacing. This gives a large-diameter graph so the comfort radius R covers only a
// LOCAL cluster of zones -> a genuine minimum-cost cover (no single station wins).
//
// testId is a difficulty/structure ladder:
//   t1 tiny (example scale) ... t10 large + adversarial (skewed costs, denser mesh).

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    int N;
    if (t <= 1) N = 9;
    else if (t == 2) N = 16;
    else if (t == 3) N = 25;
    else if (t == 4) N = 42;
    else if (t == 5) N = 60;
    else if (t == 6) N = 81;
    else if (t == 7) N = 110;
    else if (t == 8) N = 143;
    else if (t == 9) N = 169;
    else N = 200;

    int side = (int)ceil(sqrt((double)N));
    const int SP = 10; // nominal grid spacing (ski-time units)

    // jittered integer coordinates for each zone
    vector<int> X(N + 1), Y(N + 1);
    for (int i = 1; i <= N; i++) {
        int r = (i - 1) / side, c = (i - 1) % side;
        X[i] = c * SP + rnd.next(0, 4);
        Y[i] = r * SP + rnd.next(0, 4);
    }
    auto wt = [&](int a, int b) -> int {
        double dx = X[a] - X[b], dy = Y[a] - Y[b];
        int w = (int)llround(sqrt(dx * dx + dy * dy));
        if (w < 1) w = 1; if (w > 20) w = 20;
        return w;
    };

    // grid mesh edges: left neighbour + up neighbour keep the graph connected.
    set<pair<int,int>> have;
    vector<array<int,3>> edges;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        if (have.count({a, b})) return;
        have.insert({a, b});
        edges.push_back({a, b, wt(a, b)});
    };
    for (int i = 1; i <= N; i++) {
        int r = (i - 1) / side, c = (i - 1) % side;
        if (c > 0) addEdge(i, i - 1);          // left
        if (r > 0) addEdge(i, i - side);       // up
    }
    // denser meshes (even tests) add diagonal slopes; all tests add a few long slopes.
    bool dense = (t % 2 == 0);
    if (dense) {
        for (int i = 1; i <= N; i++) {
            int r = (i - 1) / side, c = (i - 1) % side;
            if (r > 0 && c > 0) addEdge(i, i - side - 1);         // up-left diagonal
            if (r > 0 && c < side - 1 && i - side + 1 <= N) addEdge(i, i - side + 1); // up-right
        }
    }
    int maxM = 4 * N + 200;
    int extra = (dense ? N / 4 : N / 8) + 3;
    for (int e = 0; e < extra && (int)edges.size() < maxM; e++) {
        int u = rnd.next(1, N), v = rnd.next(1, N);
        if (u == v) { e--; continue; }
        addEdge(u, v);
    }
    // ensure at least N-1 edges (grid guarantees connectivity already)
    int M = (int)edges.size();

    // costs
    vector<int> cost(N + 1);
    int mode = t % 3; // 0 uniform, 1 skewed (few cheap hubs), 2 wide spread
    for (int i = 1; i <= N; i++) {
        if (mode == 0) cost[i] = rnd.next(10, 100);
        else if (mode == 1) {
            if (rnd.next(0, 4) == 0) cost[i] = rnd.next(1, 12);
            else cost[i] = rnd.next(60, 200);
        } else {
            cost[i] = rnd.wnext(1, 1000, 2);
            if (cost[i] < 1) cost[i] = 1;
        }
    }

    // comfort radius: local. ~SP..1.8*SP so a station reaches direct neighbours
    // (dist ~10) but not cells two steps away (dist ~20).
    int R;
    if (t <= 1) R = 12;
    else R = 12 + (t % 4) * 3; // 12,15,18,21 cycling

    // ---- emit ----
    printf("%d %d %d\n", N, M, R);
    for (int i = 1; i <= N; i++)
        printf("%d%c", cost[i], i == N ? '\n' : ' ');
    shuffle(edges.begin(), edges.end());
    for (auto &e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
