#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: lattice of wind-tunnel sensor stations ----
    // testId 1 is tiny (example scale); grows to a large grid by testId 10.
    int side = 3 + 2 * (testId - 1);          // 3, 5, 7, ..., 21
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // weight model: most channels are EXPENSIVE (slow), a random subset are FAST.
    // The fast subgraph rarely spans the tunnel, so the fastest route threads a few
    // narrow fast corridors -- powering down corridor stations forces slow detours.
    int fastHi = 4 + (testId % 3) * 2;        // 4..8   fast-channel ceiling
    int slowLo = 25 + (testId % 4) * 6;       // 25..43 slow-channel floor
    int slowHi = slowLo + 22;
    double fastProb = 0.46 + 0.02 * (testId % 3); // fraction of fast channels

    vector<array<int, 3>> edges; // u, v, w

    auto weight = [&]() -> int {
        if (rnd.next(0.0, 1.0) < fastProb) return rnd.next(1, fastHi);
        return rnd.next(slowLo, slowHi);
    };

    // grid relay channels (right + down)
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i, j), node(i, j + 1), weight()});
            if (i + 1 < R) edges.push_back({node(i, j), node(i + 1, j), weight()});
        }

    int s = 1, t = n;

    // a few long-range shortcut channels add alternate routes (open-ended search),
    // but never a direct s-t channel (that would make interdiction impossible).
    int extra = testId;                        // 1..10 shortcuts
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        if ((a == s && b == t) || (a == t && b == s)) { e--; continue; }
        int w = rnd.next(1, fastHi + 3);
        edges.push_back({a, b, w});
    }

    int k = testId + 1;                        // budget of powered-down stations (2..11)

    // shuffle channel order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d %d %d\n", n, m, s, t, k);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
