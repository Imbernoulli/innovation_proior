#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder: a rectangular dig grid of artifact clusters ----
    // testId 1 tiny (example scale); grows to a large field by testId 10.
    int side = 3 + 2 * (testId - 1);          // 3,5,7,...,21
    int R = side, C = side;
    int n = R * C;                             // 9 .. 441

    auto node = [&](int i, int j) { return i * C + j + 1; };

    // grid correlations carry HEAVY strength (the bipartite backbone),
    // random chords carry LIGHTER strength but create odd cycles ->
    // the instance is non-bipartite and the balanced max-cut is NP-hard.
    int gridLo = 6 + (testId % 3) * 2;         // 6..10
    int gridHi = 14 + (testId % 4) * 2;        // 14..20
    int chordLo = 1;
    int chordHi = 3 + (testId % 3);            // 3..5

    vector<array<int,3>> edges;                // u, v, w

    // grid roads (right + down) -- heavy backbone
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            if (j + 1 < C) edges.push_back({node(i,j), node(i,j+1), rnd.next(gridLo, gridHi)});
            if (i + 1 < R) edges.push_back({node(i,j), node(i+1,j), rnd.next(gridLo, gridHi)});
        }

    int gridEdges = (int)edges.size();

    // random chords: create odd cycles so checkerboard is NOT optimal.
    // keep total chord weight a modest fraction of the backbone so the
    // reference (scrambled) partition stays well below the achievable cut.
    int chords = (int)(0.6 * gridEdges);
    // cap total edges to respect the m <= 2000 budget on large fields
    while (gridEdges + chords > 1900) chords--;
    if (chords < 0) chords = 0;

    for (int e = 0; e < chords; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        edges.push_back({a, b, rnd.next(chordLo, chordHi)});
    }

    int m = (int)edges.size();

    // shuffle edge order so index != structural position (hide the grid)
    shuffle(edges.begin(), edges.end());

    printf("%d %d\n", n, m);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
