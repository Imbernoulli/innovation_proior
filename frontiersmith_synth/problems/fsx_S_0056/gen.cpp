#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // Two id-contiguous "material families" (communities) [1..g] and [g+1..2g].
    // Each family hides an internal balanced sub-bipartition with heavy conflict
    // edges across it (a planted max-cut) plus light same-side noise. The index
    // split keeps each family whole, cutting NO intra-family conflict, so the
    // baseline is small; a good balanced assignment must DISCOVER the hidden
    // sub-bipartition inside each family -> genuine local-search problem.
    int g = 4 + (testId - 1) * 20;          // 4, 24, ..., 184
    int n = 2 * g;
    int slack = max(1, n / 10);

    int Hlo = 6, Hhi = 18;                   // heavy conflict weights (cross the hidden split)
    double pCross = 0.25;                     // density of heavy cross edges
    double pSame = 0.05;                      // density of light same-side noise edges
    int rTarget = 6;                          // intra_total / inter_total target

    vector<array<int,3>> edges;              // u, v, w
    ll intra_total = 0;

    for (int c = 0; c < 2; c++) {
        int lo = c * g + 1, hi = lo + g - 1;
        vector<int> nodes;
        for (int x = lo; x <= hi; x++) nodes.push_back(x);
        shuffle(nodes.begin(), nodes.end());
        int half = g / 2;
        vector<int> subA(nodes.begin(), nodes.begin() + half);
        vector<int> subB(nodes.begin() + half, nodes.end());
        if (subA.empty() || subB.empty()) { // g>=4 so never, but guard
            subA = {nodes[0]}; subB.assign(nodes.begin()+1, nodes.end());
        }

        auto add = [&](int u, int v, int w){ edges.push_back({u,v,w}); };

        // guarantee at least one heavy cross edge in this family
        {
            int w = rnd.next(Hlo, Hhi);
            add(subA[0], subB[0], w);
            intra_total += w;
        }
        // heavy cross edges (across the hidden sub-bipartition)
        for (int i = 0; i < (int)subA.size(); i++)
            for (int j = 0; j < (int)subB.size(); j++) {
                if (rnd.next(0.0, 1.0) < pCross) {
                    int w = rnd.next(Hlo, Hhi);
                    add(subA[i], subB[j], w);
                    intra_total += w;
                }
            }
        // light noise edges within each sub-side (obscure the planted cut)
        auto noise = [&](vector<int>& s){
            for (int i = 0; i < (int)s.size(); i++)
                for (int j = i + 1; j < (int)s.size(); j++)
                    if (rnd.next(0.0, 1.0) < pSame) {
                        int w = rnd.next(1, 3);
                        add(s[i], s[j], w);
                        intra_total += w;
                    }
        };
        noise(subA);
        noise(subB);
    }

    // ---- inter-family light edges, calibrated so intra_total/inter_total ~ rTarget ----
    // The index split cuts EXACTLY these edges -> this controls the baseline B.
    ll target_inter = max<ll>(1, intra_total / rTarget);
    ll inter_total = 0;
    // guarantee at least one inter edge (B > 0)
    {
        int a = rnd.next(1, g), b = rnd.next(g + 1, 2 * g), w = rnd.next(1, 3);
        edges.push_back({a, b, w}); inter_total += w;
    }
    ll guard = 0, guardMax = 6LL * (ll)edges.size() + 1000;
    while (inter_total < target_inter && guard < guardMax) {
        int a = rnd.next(1, g), b = rnd.next(g + 1, 2 * g), w = rnd.next(1, 3);
        edges.push_back({a, b, w});
        inter_total += w;
        guard++;
    }

    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, slack);
    for (auto& e : edges)
        printf("%d %d %d\n", e[0], e[1], e[2]);
    return 0;
}
