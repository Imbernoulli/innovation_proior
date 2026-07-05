#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder: number of racks grows with testId ----
    // testId 1 tiny (example scale), up to a medium data hall by testId 10.
    int half = 8 + 8 * (testId - 1);        // 8, 16, 24, ..., 80
    int n = 2 * half;                        // 16 .. 160 (even)

    // Planted balanced bipartition: original labels 1..half = side L,
    // labels half+1..n = side R. HEAVY couplings go BETWEEN the sides
    // (max-cut wants to split those); LIGHT noise couplings go WITHIN a side.
    double pCross = 0.10 + 0.02 * (testId % 4);   // density of cross couplings
    double pIntra = 0.03 + 0.01 * (testId % 3);   // density of intra (noise) couplings
    int crossLo = 8, crossHi = 20;                // heavy cross strengths
    int intraHi = 3 + (testId % 3);               // light intra strengths (1..intraHi)

    vector<array<int,3>> edges; // u, v, w in ORIGINAL labels

    // cross couplings (L x R): heavy
    for (int a = 1; a <= half; a++)
        for (int b = half + 1; b <= n; b++)
            if (rnd.next(0.0, 1.0) < pCross)
                edges.push_back({a, b, rnd.next(crossLo, crossHi)});

    // intra couplings within L and within R: light noise
    for (int a = 1; a <= half; a++)
        for (int b = a + 1; b <= half; b++)
            if (rnd.next(0.0, 1.0) < pIntra)
                edges.push_back({a, b, rnd.next(1, intraHi)});
    for (int a = half + 1; a <= n; a++)
        for (int b = a + 1; b <= n; b++)
            if (rnd.next(0.0, 1.0) < pIntra)
                edges.push_back({a, b, rnd.next(1, intraHi)});

    // guarantee a non-empty, connected-enough instance: ensure at least one
    // cross coupling exists so the reference bisection strength B is positive.
    if (edges.empty())
        edges.push_back({1, half + 1, rnd.next(crossLo, crossHi)});

    // relabel vertices by a random permutation so the planted bipartition is
    // NOT aligned with the index-based reference split -> B is a genuine
    // moderate baseline, leaving real headroom for search heuristics.
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) swap(perm[i], perm[rnd.next(1, i)]);

    for (auto& e : edges) { e[0] = perm[e[0]]; e[1] = perm[e[1]]; }

    // shuffle edge order too
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (auto& e : edges) printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
