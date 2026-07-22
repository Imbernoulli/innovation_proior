#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Seasonal Spectrum Assignment"   family: channel-map-interference-seasons
//
// N towers, C channels, EXACTLY K=7 seasons. M interfering tower pairs, each with
// a per-season conflict weight vector w[1..7] (0 means "no interference that
// season"). A solution assigns each tower one channel in [1,C]; any assignment is
// FEASIBLE (no properness requirement). For season s, S_s = sum of w_s(u,v) over
// all same-channel pairs. Objective: minimize F = sum of the TWO LARGEST S_s
// (sum-of-worst-two-scoring) -- the other five seasons are free to be bad.
//
// PLANTED TRAP STRUCTURE (never explained in the statement -- must be discovered
// from the data): towers are grouped into small CLIQUES of size L = C + extra
// (L > C), so by pigeonhole every clique is forced to have some same-channel
// pairs no matter how it is colored. Each clique edge is independently labelled
// either
//   HOT   -- weight WH in BOTH of this test's two "hot" seasons h1,h2 (chosen at
//            random per test), 0 elsewhere. Per-edge total weight = 2*WH.
//   MILD  -- weight WD in exactly ONE of the five non-hot seasons (chosen at
//            random per edge), 0 elsewhere. Per-edge total weight = WD.
// with 2*WH < WD, i.e. a hot edge looks CHEAPER than a mild edge if you only look
// at each edge's own total weight summed over all seasons. A solver that colors
// cliques to minimize TOTAL weighted same-channel conflict (ignoring which
// specific seasons the weight lands in) will happily let hot edges collide
// (they look "cheap") while fighting hard to avoid the "expensive-looking" mild
// edges. But every hot collision stacks onto the SAME two seasons h1,h2, which
// become (and stay) the binding top-2 pair, while mild collisions spread their
// damage across five different season buckets and rarely bind at all. The
// correct move is the opposite of the "cheap edge" heuristic: always sacrifice
// mild edges, and additionally balance which of the five mild seasons absorbs
// each sacrifice so no single one grows large enough to become binding either.
//
// gen also emits: NEEDLE tests (a sea of tiny background weights hiding one real
// planted cluster), a PLANTED test (hot edges form only a sparse matching inside
// each clique, i.e. exactly C-colorable -> a smart solver can drive hot damage to
// the true combinatorial floor), and scale/adversarial tests filling the size
// envelope. Tower ids are shuffled so id-adjacency never leaks cluster structure
// and the round-robin baseline cannot accidentally align with it.
// -----------------------------------------------------------------------------

struct Edge { int u, v; int w[8]; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- per-test parameters -------------------------------------------------
    int N, C, Lextra;
    double pHot;
    int WH, WD;
    bool needle = false, planted = false;
    double bgFrac; // fraction of extra background (inter-cluster) edges, relative to N

    switch (testId) {
        case 1:  N = 12;  C = 2; Lextra = 1; pHot = 0.35; WH = 15; WD = 80;  bgFrac = 0.4; break;
        case 2:  N = 28;  C = 3; Lextra = 1; pHot = 0.35; WH = 15; WD = 85;  bgFrac = 0.5; break;
        case 3:  N = 50;  C = 3; Lextra = 2; pHot = 0.38; WH = 18; WD = 100; bgFrac = 0.3; break;
        case 4:  N = 72;  C = 4; Lextra = 2; pHot = 0.38; WH = 16; WD = 105; bgFrac = 0.3; break;
        case 5:  N = 96;  C = 3; Lextra = 2; pHot = 0.42; WH = 14; WD = 110; bgFrac = 0.5; break;
        case 6:  N = 160; C = 4; Lextra = 2; pHot = 0.38; WH = 18; WD = 100; bgFrac = 0.2; needle = true; break;
        case 7:  N = 180; C = 3; Lextra = 2; pHot = 0.42; WH = 15; WD = 105; bgFrac = 0.3; planted = true; break;
        case 8:  N = 260; C = 4; Lextra = 2; pHot = 0.38; WH = 16; WD = 102; bgFrac = 0.3; needle = true; break;
        case 9:  N = 360; C = 5; Lextra = 2; pHot = 0.40; WH = 15; WD = 106; bgFrac = 0.25; break;
        case 10: N = 460; C = 5; Lextra = 3; pHot = 0.38; WH = 16; WD = 110; bgFrac = 0.3; planted = true; break;
        default: N = 20;  C = 3; Lextra = 1; pHot = 0.35; WH = 15; WD = 80;  bgFrac = 0.4; break;
    }

    int L = C + Lextra; // clique size, always > C -> pigeonhole guarantees collisions

    // Choose the two "hot" seasons for this test (1-indexed, 1..7), and keep the
    // remaining 5 as "mild" seasons.
    int h1 = rnd.next(1, 7);
    int h2 = rnd.next(1, 7);
    while (h2 == h1) h2 = rnd.next(1, 7);
    vector<int> mildSeasons;
    for (int s = 1; s <= 7; s++) if (s != h1 && s != h2) mildSeasons.push_back(s);

    // Random permutation of tower ids 1..N so structure never correlates with id order.
    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i + 1;
    for (int i = N - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);
    int nextSlot = 0;
    auto takeTower = [&]() -> int { return perm[nextSlot++]; };

    vector<Edge> edges;
    auto addEdge = [&](int u, int v, int seasonMask /*bit0=h1 hot, otherwise mild season id*/,
                        bool isHot, int weight, int mildSeason) {
        Edge e; e.u = u; e.v = v;
        for (int s = 1; s <= 7; s++) e.w[s] = 0;
        if (isHot) { e.w[h1] = weight; e.w[h2] = weight; }
        else { e.w[mildSeason] = weight; }
        edges.push_back(e);
    };

    // ---- clique gadgets --------------------------------------------------
    // needle tests: a smaller SHARE of towers carry the real weighted structure
    // (more of the instance is background), but the total planted signal still
    // scales with N so it is not swamped by the noise flood below.
    int nClusterTowersBudget = needle ? (int)(N * 0.30) : (int)(N * 0.62);
    vector<vector<int>> clusters;
    while (nextSlot + L <= N && (int)clusters.size() * L < nClusterTowersBudget) {
        vector<int> members;
        for (int i = 0; i < L; i++) members.push_back(takeTower());
        clusters.push_back(members);
    }

    for (auto &members : clusters) {
        int Lc = members.size();
        // Planted test: hot edges form only a sparse matching (<=1 hot edge per
        // vertex) inside the clique, i.e. the hot subgraph alone is trivially
        // C-colorable (C>=2), so a solver that specifically protects hot edges can
        // drive their damage to exactly the clique's forced floor. Non-planted
        // tests instead give hot edges arbitrary (denser) placement, which still
        // permits avoiding them entirely as long as the search deliberately steers
        // away from them -- this is the general case a real search must handle.
        vector<int> hotDeg(Lc, 0);
        for (int a = 0; a < Lc; a++) {
            for (int b = a + 1; b < Lc; b++) {
                bool isHot;
                if (planted) {
                    isHot = (hotDeg[a] == 0 && hotDeg[b] == 0 && rnd.next(0.0, 1.0) < pHot);
                } else {
                    isHot = rnd.next(0.0, 1.0) < pHot;
                }
                if (isHot) { hotDeg[a]++; hotDeg[b]++; }
                int ms = mildSeasons[rnd.next(0, (int)mildSeasons.size() - 1)];
                int w = isHot ? WH : WD;
                // small jitter so weights are not perfectly uniform (avoids trivial
                // "all edges identical" degeneracy) but keeps the 2*WH < WD trap intact.
                w = max(1, w + rnd.next(-w / 6, w / 6));
                addEdge(members[a], members[b], 0, isHot, w, ms);
            }
        }
    }

    // ---- background / noise edges -----------------------------------------
    int bgCount = (int)(N * bgFrac);
    int noiseW = needle ? rnd.next(1, 4) : -1;
    for (int i = 0; i < bgCount; i++) {
        int u = rnd.next(1, N), v = rnd.next(1, N);
        if (u == v) continue;
        if (u > v) swap(u, v);
        int s = rnd.next(1, 7);
        int w = needle ? rnd.next(1, 5) : rnd.next(3, 25);
        Edge e; e.u = u; e.v = v;
        for (int t = 1; t <= 7; t++) e.w[t] = 0;
        e.w[s] = w;
        edges.push_back(e);
    }

    // needle tests: flood in a large number of very low-weight random edges over
    // the WHOLE tower set so the one real cluster must be found amid noise.
    if (needle) {
        int floodCount = min(N * 2, 500);
        for (int i = 0; i < floodCount; i++) {
            int u = rnd.next(1, N), v = rnd.next(1, N);
            if (u == v) continue;
            if (u > v) swap(u, v);
            int s = rnd.next(1, 7);
            int w = rnd.next(1, 2);
            Edge e; e.u = u; e.v = v;
            for (int t = 1; t <= 7; t++) e.w[t] = 0;
            e.w[s] = w;
            edges.push_back(e);
        }
    }

    // merge duplicate (u,v) pairs (sum their weight vectors) so each unordered
    // pair appears at most once, as promised by the statement.
    map<pair<int,int>, array<int,8>> merged;
    for (auto &e : edges) {
        int u = e.u, v = e.v;
        if (u > v) swap(u, v);
        auto key = make_pair(u, v);
        auto it = merged.find(key);
        if (it == merged.end()) {
            array<int,8> arr{};
            for (int s = 1; s <= 7; s++) arr[s] = e.w[s];
            merged[key] = arr;
        } else {
            for (int s = 1; s <= 7; s++) it->second[s] += e.w[s];
        }
    }

    vector<pair<pair<int,int>, array<int,8>>> finalEdges(merged.begin(), merged.end());
    // shuffle output line order
    for (int i = (int)finalEdges.size() - 1; i > 0; i--)
        swap(finalEdges[i], finalEdges[rnd.next(0, i)]);

    int M = (int)finalEdges.size();
    printf("%d %d %d\n", N, C, M);
    for (auto &pr : finalEdges) {
        int u = pr.first.first, v = pr.first.second;
        printf("%d %d", u, v);
        for (int s = 1; s <= 7; s++) printf(" %d", pr.second[s]);
        printf("\n");
    }
    return 0;
}
