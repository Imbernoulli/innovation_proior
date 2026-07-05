#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Firewatch: minimum-cost R-covering (weighted R-dominating set) on a forest-cell graph.
//
// testId is a difficulty/structure ladder:
//   testId 1  -> tiny (example scale) so the worked example makes sense.
//   testId 10 -> large, sparse-but-nontrivial adjacency with skewed costs.
// We keep the average degree modest and R small so that the all-towers baseline B is a
// genuinely competitive reference (no solution can beat it by more than ~10x -> no cap
// pile-up), while cost skew makes cost-aware and coverage-aware heuristics diverge.

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder ----
    int N;
    if (testId == 1) N = 40;                 // tiny, example scale
    else             N = 1400 * (testId - 1); // 1400, 2800, ..., 12600

    // ---- radius: mostly 1, a couple of R=2 tests for structural variety ----
    int R = (testId == 4 || testId == 8) ? 2 : 1;

    // ---- average degree (kept modest to keep coverage balls small) ----
    // With R=2 the ball grows, so use a sparser graph on those tests.
    double avgDeg;
    if (R == 2) avgDeg = 2.4 + 0.1 * (testId % 3);   // ~2.4..2.6
    else        avgDeg = 3.0 + 0.5 * (testId % 3);   // ~3.0..4.0

    long long targetM = (long long)llround(avgDeg * N / 2.0);
    if (targetM < N - 1) targetM = N - 1;            // enough for a spanning-ish structure

    // ---- cost model: alternate uniform vs skewed (few cheap cells) ----
    bool skewed = (testId % 2 == 0);
    vector<int> cost(N + 1);
    for (int v = 1; v <= N; v++) {
        if (skewed) {
            if (rnd.next(0.0, 1.0) < 0.15) cost[v] = rnd.next(1, 3);    // rare cheap cell
            else                           cost[v] = rnd.next(12, 25);  // most cells costly
        } else {
            cost[v] = rnd.next(1, 25);                                  // uniform
        }
    }

    // ---- build edges ----
    // Start with a random spanning path/tree so the graph is reasonably connected and R=1
    // balls actually overlap spatially (a corridor network, not isolated islands), then add
    // random extra corridors up to targetM. Dedup with a hash set.
    set<pair<int,int>> have;
    auto key = [](int a, int b) { return a < b ? make_pair(a, b) : make_pair(b, a); };

    vector<int> perm(N);
    for (int i = 0; i < N; i++) perm[i] = i + 1;
    shuffle(perm.begin(), perm.end());
    // random tree: attach each new node to a random earlier node
    for (int i = 1; i < N; i++) {
        int a = perm[i];
        int b = perm[rnd.next(0, i - 1)];
        have.insert(key(a, b));
    }

    int guard = 0;
    while ((long long)have.size() < targetM && guard < 40 * (int)targetM + 1000) {
        guard++;
        int a = rnd.next(1, N);
        int b = rnd.next(1, N);
        if (a == b) continue;
        have.insert(key(a, b));
    }

    vector<pair<int,int>> edges(have.begin(), have.end());
    shuffle(edges.begin(), edges.end());
    int M = (int)edges.size();

    // ---- emit ----
    printf("%d %d %d\n", N, M, R);
    for (int v = 1; v <= N; v++) printf("%d%c", cost[v], v == N ? '\n' : ' ');
    for (auto& e : edges) printf("%d %d\n", e.first, e.second);
    return 0;
}
