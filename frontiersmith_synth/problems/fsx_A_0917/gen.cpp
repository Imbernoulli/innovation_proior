#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Terrace Irrigation Retrofit"   family: matroid-intersection-packing
//
// PLANTED TRAP (present on every test, dominant on >=3): for each of R "shipment
// rounds" we mint two fresh premium batches P1,P2 (cap = m each) and lay down m
// independent 2-pair gadgets:
//   pair A (nodes u1,v1): a parallel edge in P1 weight H, and a parallel edge in P2
//                          weight M   (H > M; picking BOTH is a cycle -- only one fires)
//   pair B (nodes u2,v2): a single edge in P1 weight L                (L < M < H)
// with M + L > H. The batch-cap-oblivious "sort by weight, add if independent"
// greedy takes the globally-heaviest H edge on every pair-A first (exhausting P1
// before ever trying pair B), scoring H per gadget; the true weighted matroid-
// intersection optimum instead spends P2 on pair A (M) and saves P1 for pair B (L),
// scoring M+L > H per gadget. This is exactly the classical "two-matroid greedy is
// not optimal" counterexample (it specializes to weighted bipartite matching), here
// realized as parallel edges (graphic matroid) competing for a shared batch cap
// (partition matroid).
//
// A separate, always-batch-1 "reservoir main line" component supplies the checker's
// baseline B (a simple path, deterministic weight) so trivial reproduces B exactly.
// -----------------------------------------------------------------------------

struct Edge { int u, v; ll w; int b; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    vector<Edge> edges;
    int nextNode = 1;
    int K = 1; // batch 1 = standard stock, unlimited

    // ---- test-id ladder: (m per round, R rounds, baseline nodes) ----
    struct Ladder { ll m, R, Nb; ll base, d; };
    Ladder L;
    if (testId == 1) {
        L = {1, 1, 3, 10, 1};
    } else {
        static const ll ms[11]  = {0, 0, 4, 10, 20, 18, 27, 39, 36, 48, 64};
        static const ll Rs[11]  = {0, 0, 1,  1,  1,  2,  2,  2,  3,  3,  3};
        static const ll Nbs[11] = {0, 0, 8, 10, 12, 14, 16, 18, 22, 26, 30};
        L.m = ms[testId];
        L.R = Rs[testId];
        L.Nb = Nbs[testId];
        L.base = 100 + 5 * testId;
        L.d = 5;
    }

    ll T = L.m * L.R;
    ll targetB = (testId == 1) ? 10 : 50 * T;

    // ---- trap rounds ----
    for (int r = 0; r < L.R; r++) {
        int P1 = ++K, P2 = ++K;
        for (int i = 0; i < L.m; i++) {
            int u1 = nextNode++, v1 = nextNode++;
            int u2 = nextNode++, v2 = nextNode++;
            // mild per-instance jitter that preserves H>M>L and M+L>H strictly
            ll jitter = rnd.next(0, (int)(L.d / 2));
            ll H = L.base + 2 * L.d + jitter;
            ll M = L.base + L.d + jitter;
            ll Lo = L.base + jitter;
            edges.push_back({u1, v1, H, P1});
            edges.push_back({u1, v1, M, P2});
            edges.push_back({u2, v2, Lo, P1});
        }
    }

    // ---- baseline "reservoir main line": a simple path of batch-1 edges, exact weight ----
    ll Nb = max((ll)2, L.Nb);
    ll Wb = max((ll)1, targetB / (Nb - 1));
    vector<int> mainNodes;
    for (ll i = 0; i < Nb; i++) mainNodes.push_back(nextNode++);
    for (ll i = 0; i + 1 < Nb; i++)
        edges.push_back({mainNodes[i], mainNodes[i + 1], Wb, 1});

    // ---- a little unrelated filler noise: cheap random standard-batch edges in a
    //      disjoint component (they add a small, harmless amount to B and F alike,
    //      since batch 1 never binds and neither greedy nor strong is affected
    //      differently by picking them) ----
    int filler = (int)min<ll>(20, 2 + Nb / 2);
    vector<int> fillerNodes;
    for (int i = 0; i < filler; i++) fillerNodes.push_back(nextNode++);
    for (int i = 0; i + 1 < filler; i++) {
        ll w = 1 + rnd.next(0, (int)max((ll)1, Wb / 5));
        edges.push_back({fillerNodes[i], fillerNodes[i + 1], w, 1});
    }

    int V = nextNode - 1;
    int E = (int)edges.size();

    // shuffle output order (Fisher-Yates via rnd) -- pure cosmetic/no positional cues
    for (int i = E - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(edges[i], edges[j]);
    }

    vector<ll> cap(K + 1, 0);
    cap[1] = E + 10;
    {
        int idx = 2;
        for (int r = 0; r < L.R; r++) {
            cap[idx++] = L.m;
            cap[idx++] = L.m;
        }
    }

    printf("%d %d %d\n", V, E, K);
    for (int b = 1; b <= K; b++) printf("%lld%c", cap[b], b == K ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d %lld %d\n", e.u, e.v, e.w, e.b);
    return 0;
}
