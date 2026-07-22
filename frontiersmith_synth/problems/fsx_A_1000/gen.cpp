#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Generator for "Guild Forecast Tournament: Liquidity Allocation Under the
// Market-Scoring Rule"  (family: lmsr-liquidity-allocator)
//
// N=100 questions every test. Each question q is stamped with a category that
// controls how its printed trader roster behaves once a solver's beta_q is
// replayed against it:
//
//   AGREE  -- every trader's belief clusters tightly around the true outcome.
//             The crowd already agrees, so the price converges to nearly the
//             same place at almost any beta_q>0: pouring extra liquidity here
//             buys essentially nothing (the classic "uniform / more-liquidity
//             where more traders" trap -- lots of traffic, zero marginal
//             value).
//   TRAP   -- exactly two large, DISAGREEING, well-funded traders open the
//             book (first a confident WRONG faction, then a confident RIGHT
//             faction with slightly less capital), followed by a tail of
//             small noise traders. At beta_q -> 0 the LAST large mover (or
//             worse, a late noise trader) dictates the final price almost
//             for free; at beta_q -> large nobody can afford to move the
//             price off 0.5; only a beta_q sized so the RIGHT faction's
//             budget is (just) enough to overpower the WRONG faction's
//             earlier move -- but the WRONG faction's budget is NOT enough to
//             fully re-overpower back -- lands the price near the truth.
//             This is where the marginal value of beta_q is highest, and
//             where it is also non-monotone in beta_q (small beta rewards
//             "whoever moves last", not "who has more money").
//   NEEDLE -- one single question (present only from testId>=6) where BOTH
//             factions carry an order of magnitude more capital than every
//             other question combined needs; getting its beta_q right is
//             worth as much as several ordinary questions, so an allocator
//             that spreads liquidity "fairly" leaves enormous value on the
//             table on this one line.
//
// Structural ladder: testId grows M_q (roster depth) and the TRAP fraction,
// and testId>=6 also plants exactly one NEEDLE question -- adversarial-ness
// and input size both grow monotonically with testId while N stays fixed at
// the tournament's fixed size (100 questions).
// -----------------------------------------------------------------------------

static int clampInt(int v, int lo, int hi) { return max(lo, min(hi, v)); }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int N = 100;
    const long long B = 10000;
    const double LAMBDA = 0.005;

    double fracTrap = min(0.50, 0.15 + 0.04 * testId);
    bool haveNeedle = (testId >= 6);
    int needleIdx = haveNeedle ? (N - 1) : -1;

    int M_lo = 2 + testId;           // 3 .. 12
    int M_hi = 5 + 3 * testId;       // 8 .. 35

    printf("%d %lld %.6f\n", N, B, LAMBDA);

    for (int q = 0; q < N; q++) {
        string kind;
        if (q == needleIdx) kind = "needle";
        else if (rnd.next(0.0, 1.0) < fracTrap) kind = "trap";
        else kind = "agree";

        int outcome = rnd.next(0, 1);
        vector<pair<int,int>> traders; // (belief percent, budget)

        if (kind == "agree") {
            int targetPct = (outcome == 1) ? rnd.next(75, 93) : rnd.next(7, 25);
            int mHiAgree = max(M_lo + 1, M_lo + (M_hi - M_lo) / 2);
            int M = rnd.next(M_lo, mHiAgree);
            for (int i = 0; i < M; i++) {
                int bp = clampInt(targetPct + rnd.next(-6, 6), 2, 98);
                int budget = rnd.next(30, 250);
                traders.push_back({bp, budget});
            }
        } else if (kind == "trap") {
            int wrongPct   = (outcome == 0) ? rnd.next(85, 95) : rnd.next(5, 15);
            int correctPct = (outcome == 1) ? rnd.next(85, 95) : rnd.next(5, 15);
            traders.push_back({wrongPct, rnd.next(1200, 2600)});
            traders.push_back({correctPct, rnd.next(900, 2000)});
            int M = rnd.next(M_lo + 5, M_hi + 14);
            for (int i = 0; i < M; i++) {
                int bp = rnd.next(5, 95);
                int budget = rnd.next(30, 300);
                traders.push_back({bp, budget});
            }
        } else { // needle
            int wrongPct   = (outcome == 0) ? rnd.next(90, 97) : rnd.next(3, 10);
            int correctPct = (outcome == 1) ? rnd.next(90, 97) : rnd.next(3, 10);
            traders.push_back({wrongPct, rnd.next(6000, 9000)});
            traders.push_back({correctPct, rnd.next(5000, 8000)});
            int M = rnd.next(M_lo, M_hi);
            for (int i = 0; i < M; i++) {
                int bp = rnd.next(5, 95);
                int budget = rnd.next(30, 300);
                traders.push_back({bp, budget});
            }
        }

        int Mq = (int)traders.size();
        printf("%d %d\n", Mq, outcome);
        for (auto& tr : traders) printf("%d %d\n", tr.first, tr.second);
    }

    return 0;
}
