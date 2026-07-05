#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: harbor yard grid grows with testId ----
    // testId 1 is tiny (example scale); grows to a large medium instance by testId 10.
    int side = 5 + 3 * (testId - 1);           // 5, 8, 11, ..., 32
    int R = side, C = side;
    int n = R * C;

    auto node = [&](int i, int j) { return i * C + j + 1; };

    vector<array<int,3>> edges; // u v w

    // grid haulage lanes (right + down) with small travel times
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i,j), node(i,j+1), rnd.next(1,5)});
            if (i + 1 < R) edges.push_back({node(i,j), node(i+1,j), rnd.next(1,5)});
        }

    // a few express corridors add long-range shortcuts (richer distance structure)
    int extra = 2 * testId;
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, rnd.next(1, 5)});
    }
    int m = (int)edges.size();

    // ---- demand blocks: a random subset of yard blocks ----
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());
    // demand fraction shifts across the ladder to vary coverage difficulty
    double frac = 0.28 + 0.02 * (testId % 4);   // 0.28..0.34
    int D = max(1, (int)(n * frac));
    if (D > n) D = n;
    vector<int> demand(perm.begin(), perm.begin() + D);

    // ---- candidate stations ----
    // (a) a LOCAL station at every demand block (small radius) -> guarantees feasibility
    //     because a station covers its own block (distance 0 <= radius).
    // (b) HUB stations at random blocks with larger radius -> cover many demands cheaply,
    //     creating the greedy-vs-local-search trade-off.
    vector<array<int,3>> stations; // node cost radius
    for (int d : demand) {
        int r = rnd.next(1, 3);
        int c = rnd.next(6, 12);
        stations.push_back({d, c, r});
    }
    int H = max(3, n / 22);
    for (int h = 0; h < H; h++) {
        int nd = rnd.next(1, n);
        int r = rnd.next(4, 9);
        // hub cost scales mildly with radius so it is efficient but not free
        int c = rnd.next(22, 40) + r;
        stations.push_back({nd, c, r});
    }

    // shuffle station order so index != structural role
    shuffle(stations.begin(), stations.end());
    int P = (int)stations.size();

    // ---- emit ----
    printf("%d %d\n", n, m);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    printf("%d\n", D);
    for (int i = 0; i < D; i++) printf("%d%c", demand[i], i + 1 < D ? ' ' : '\n');
    printf("%d\n", P);
    for (auto& st : stations) printf("%d %d %d\n", st[0], st[1], st[2]);
    return 0;
}
