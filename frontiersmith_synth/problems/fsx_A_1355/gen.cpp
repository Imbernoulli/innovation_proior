#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Everyone Guesses, One Must Be Right"  (generator)  family: hat-guessing-strategy
//
// N players, K hat colours (0..K-1), full mutual visibility (everyone sees every
// OTHER player's colour, nobody sees their own). Each player i carries an
// information weight w_i (its guess, if correct, is worth w_i). The strategy
// designer must output, for every player, a full guess table over all K^(N-1)
// combinations of colours they could see. The checker brute-forces ALL K^N
// colour assignments and scores the WORST-CASE total weight of correct guesses
// (adversarial-assignment).
//
// PLANTED / TRAP / NEEDLE structure (never labelled in the input -- solvers must
// discover it from the raw weight list):
//   Most players get a small "filler" weight (roughly uniform noise). A handful
//   of players -- those at positions N, N-K, N-2K, ... (spaced by K, anchored at
//   the END of the list) -- carry a "chain" weight that grows geometrically
//   (needle: a few high-value items amid noise). Two consequences, BOTH
//   engineered on purpose:
//     (a) All chain positions share the SAME index residue ((index-1) mod K is
//         the same constant for every position N-jK) -- so any strategy that
//         partitions players into K groups purely by INDEX ROUND-ROBIN dumps
//         every valuable player into one group, starving the other K-1 groups
//         (bad baseline).
//     (b) The chain values appear in ASCENDING order as you scan the players in
//         index order, and every chain item is preceded by many filler items
//         (index order IS arrival order for an online balancer) -- so a
//         strategy that greedily balances players ONE AT A TIME IN GIVEN ORDER
//         has already locked in a near-even filler-only split across all K
//         groups by the time the first (smallest) big item arrives, and each
//         subsequent, larger item then gets dumped onto whatever group is
//         momentarily lightest -- leaving one group permanently starved (trap:
//         greedy-in-given-order lands far from optimal).
//   A strategy that instead SORTS players by weight descending before greedily
//   balancing (classic exchange-argument insight) places the heaviest, most
//   informative guessers first while every group is still empty, and achieves a
//   near-perfectly balanced partition regardless of index or arrival order.
//
// Output:  N K   then one line   w_1 w_2 ... w_N
// -----------------------------------------------------------------------------

struct Cfg { int N, K, flo, fhi, base, ratio; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    static const Cfg cfgs[10] = {
        {3, 2, 1, 3, 6, 3},
        {4, 2, 1, 3, 6, 3},
        {5, 2, 1, 4, 7, 3},
        {5, 3, 1, 4, 7, 3},
        {6, 2, 1, 4, 7, 3},
        {6, 3, 1, 4, 7, 3},
        {7, 3, 1, 5, 8, 3},
        {8, 3, 1, 5, 8, 3},
        {8, 4, 1, 5, 9, 3},
        {9, 4, 1, 5, 9, 3},
    };
    Cfg c = cfgs[testId - 1];
    int N = c.N, K = c.K;

    vector<ll> w(N + 1);
    for (int i = 1; i <= N; i++) w[i] = c.flo + rnd.next(0, c.fhi - c.flo);

    // Chain positions anchored at the END, spaced by K (share one residue
    // under index-round-robin -- the trivial reference clusters them all
    // into one group -- and leave as much filler as possible BEFORE the
    // first chain item, so an in-order online balancer commits its bins to
    // filler weight before any high-value guesser arrives).
    vector<int> chainPos;
    for (int p = N; p >= 1; p -= K) chainPos.push_back(p);
    sort(chainPos.begin(), chainPos.end());
    ll val = c.base;
    for (int p : chainPos) {
        int jitter = max(1, (int)(val / 10));
        w[p] = val + rnd.next(0, jitter);
        val *= c.ratio;
    }

    printf("%d %d\n", N, K);
    for (int i = 1; i <= N; i++) printf("%lld%c", w[i], i == N ? '\n' : ' ');
    return 0;
}
