#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Balance-constrained max-cut ("Aurora Station two-habitat crew split").
// Structure ladder: testId 1 is tiny (example scale); n and the edge density grow
// toward testId 10. A heavy "friction chain" (i,i+1) is always planted so the
// contiguous-split baseline the checker measures is deliberately weak, giving the
// heuristics genuine head-room; random + heavy-tailed extra edges make the max-cut hard.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n;
    if (testId == 1) n = 6;
    else n = 30 * testId;                 // 60, 90, ..., 300
    if (n > 400) n = 400;

    // headcount slack: strict for tiny tests, looser (but still "roughly balanced") later
    int L = max(1, n / 10);

    // weight model: chain records are moderate; extras are mostly light with a
    // heavy tail so greedy and swap-based search diverge.
    int chainLo = 5, chainHi = 15;
    int lightLo = 1, lightHi = 8;
    int heavyLo = 20, heavyHi = 40;
    double heavyProb = 0.10 + 0.02 * (testId % 3);   // fraction of heavy extras

    vector<array<int,3>> edges;           // u, v, w

    // planted friction chain (guarantees B > 0 for the contiguous baseline)
    for (int i = 1; i < n; i++)
        edges.push_back({i, i + 1, rnd.next(chainLo, chainHi)});

    // random extra friction records
    int extra;
    if (testId == 1) extra = 3;
    else extra = min(4000 - (n - 1), 6 * n);
    for (int e = 0; e < extra; e++) {
        int a = rnd.next(1, n), b = rnd.next(1, n);
        if (a == b) { e--; continue; }
        int w;
        if (rnd.next(0.0, 1.0) < heavyProb) w = rnd.next(heavyLo, heavyHi);
        else                                w = rnd.next(lightLo, lightHi);
        edges.push_back({a, b, w});
    }

    // shuffle so edge order carries no positional information
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, L);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
