#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Win a two-sided idea war on a network"  (generator)  family: competitive-front-seeding
//
// The graph is a disjoint union of R independent ARENAS. Arena r has:
//   - 3 "control" nodes c0,c1,c2 (open -- available for the solver to seed A),
//   - 1 "incumbent" node H (pre-seeded with the rival contagion B),
//   - d_r "market" nodes, each wired to H and to exactly TWO of {c0,c1,c2}
//     (its "pair-type" 01 / 02 / 12, drawn unevenly so the three pairs carry
//     different total value -- WHICH pair to back is a real decision, not a
//     coin flip).
//
// PLANTED STRUCTURE (never labelled -- must be read off the edge list): every
// market node has degree exactly 3, and its rival vote (from H) is always
// present. Seeding ONE control node gives a market node only 1 A-vote against
// H's 1 B-vote -- a tie, forever (majority-threshold: strictly more, or no
// change). Seeding the matching PAIR of control nodes for a market node's
// type gives it 2 A-votes vs 1 B-vote and it converts in round 1. So a lone
// "counter-seed" per arena (the classic influence-maximisation move: put your
// strongest voice next to the incumbent) converts NOTHING, while paying for
// the right ADJACENT pair converts every market node of that pair's type in
// one round -- and paying for all three controls converts the whole arena.
// This is scale-free (direct 1-hop reach, independent of d_r), so it holds
// on both the tiny example and the largest generated tests.
//
// Arenas are randomised into "premium" (few, high-value market nodes) and
// "bulk" (many, low-value market nodes) so raw per-node value does not
// simply track total arena worth -- a solver must multiply pair-count by
// per-node value to see which arenas (and which pair inside them) are worth
// unlocking. Budget K is roughly R/2 (< R), so a single "one seed per arena"
// sweep can never afford even one pair, guaranteeing the trap bites broadly.
//
// Output:  N M K S \n  N values \n  M edges (u v) \n  S rival ids
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;   // 0..1 difficulty ladder

    int R = (int)llround(6 + f * 74.0);              // 6..80 arenas (scale via COUNT)
    int unlockBudget = max(2, R / 4);
    int K = 2 * unlockBudget;                          // even, K < R for R>=3

    // arena SIZE stays modest regardless of scale -- otherwise a fully-unlocked
    // arena's prize dwarfs any reasonable baseline by more than the 10x score cap
    int premLoD = 6, premHiD = 12;
    int bulkLoD = 14, bulkHiD = 28;

    vector<ll> val;               // 1-indexed via val[0] unused
    val.push_back(0);
    vector<pair<int,int>> edgeVec;
    vector<int> rivalIds;

    int idCounter = 1;
    for (int r = 1; r <= R; r++){
        int c0 = idCounter++, c1 = idCounter++, c2 = idCounter++;
        int H  = idCounter++;
        val.resize(idCounter, 0);          // controls + hub have value 0
        rivalIds.push_back(H);

        bool premium = (rnd.next(0, 99) < 40);
        int d = premium ? rnd.next(premLoD, premHiD) : rnd.next(bulkLoD, bulkHiD);
        ll lo = premium ? 20 : 3, hi = premium ? 60 : 9;

        // uneven pair-type weights (which of the 3 pairs is most "popular")
        int w0 = 1 + rnd.next(0, 8), w1 = 1 + rnd.next(0, 8), w2 = 1 + rnd.next(0, 8);
        int wsum = w0 + w1 + w2;

        for (int j = 0; j < d; j++){
            int id = idCounter++;
            val.resize(idCounter, 0);
            val[id] = lo + rnd.next(0, (int)(hi - lo));
            int pick = rnd.next(0, wsum - 1);
            int type = (pick < w0) ? 0 : (pick < w0 + w1 ? 1 : 2);
            int a = (type == 0) ? c0 : (type == 1) ? c0 : c1;
            int b = (type == 0) ? c1 : (type == 1) ? c2 : c2;
            edgeVec.push_back({id, a});
            edgeVec.push_back({id, b});
            edgeVec.push_back({id, H});
        }
    }
    int N = idCounter - 1;
    // deterministic shuffle of edge listing order only (does not change the graph)
    for (int i = (int)edgeVec.size() - 1; i > 0; i--) swap(edgeVec[i], edgeVec[rnd.next(0, i)]);
    int M = (int)edgeVec.size();
    int S = (int)rivalIds.size();

    printf("%d %d %d %d\n", N, M, K, S);
    for (int i = 1; i <= N; i++) printf("%lld%c", val[i], i == N ? '\n' : ' ');
    for (auto &e : edgeVec) printf("%d %d\n", e.first, e.second);
    for (int i = 0; i < S; i++) printf("%d%c", rivalIds[i], i == S - 1 ? '\n' : ' ');
    return 0;
}
