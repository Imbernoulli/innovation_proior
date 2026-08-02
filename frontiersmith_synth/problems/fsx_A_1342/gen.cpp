#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Pairing Guard on a Contested Board"  (family: maker-breaker-pairing)
//
// Board of n cells, m winning lines (2..6 cells each, weight w). Aggressor and
// Guard alternately claim cells; Guard commits in ADVANCE to a pairing (matching)
// of cells, spending a zone-distance budget K, and reacts to the Aggressor's
// fixed, given claim order by taking a matched partner (or, if none, the
// lowest-indexed free cell). A line is SAFE iff the Guard ends up owning at
// least one of its cells.
//
// PLANTED STRUCTURE (checker never sees these labels):
//   "Plain" lines: fresh, mutually cell-disjoint lines. Any reasonable pairing
//   protects them, so they contribute equally to every solver -- they exist to
//   keep the ladder honest (trivial << greedy, everyone benefits from them) and
//   to fill the board out realistically.
//
//   "Hub gadgets" (testId >= 4, 7 of the 10 tests, escalating count): three fresh
//   cells u,v,d per gadget, zones z_u=0, z_v=5, z_d=0 (fixed, NOT randomised --
//   this is the load-bearing part of the trap and must never depend on rnd).
//     decoy line   L1 = {u, v, d}   weight w1   (listed in this exact order)
//     hub line     L2 = {u, v}      weight w2 < w1
//   cost(u,d)=1, cost(u,v)=cost(v,d)=6.
//   A per-line myopic packer (sort lines by weight desc, pair the two cheapest
//   still-free cells of the current line) processes L1 first (w1 is the larger
//   weight) and locks u to the decoy d for just 1 budget unit -- because d is
//   *cheap*, not because it's useful. That strands v: by the time L2 is
//   considered, u is gone, so L2 can never be paired and is lost outright. The
//   insight is to recognise that the single pair (u,v) is a shared "hub" that
//   sits inside BOTH lines (an explicit involution covering two winning lines
//   with one couple) -- worth w1+w2, strictly more than the w1 a myopic packer
//   settles for -- regardless of the random weight draw.
//
// Budget K is deliberately generous for the plain lines (a solver that reasons
// about the board at all can always afford every plain line) and exactly funds
// every gadget's hub pair, so the score gap is attributable to the STRUCTURAL
// choice (which pair to commit to), not to budget starvation.
//
// The Aggressor's claim order is a fixed permutation, sorted by descending
// per-cell "threat weight" (sum of weights of lines through that cell), ties
// broken by ascending index -- an aggressive, non-adaptive, but well-informed
// plan that is handed to the solver verbatim.
// -----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int nPlain = 10 + (int)llround(f * 60.0);              // 10 .. 70 plain lines
    int nGadget = (testId <= 3) ? 0 : (testId - 3) * 4;      // 0,0,0,4,8,12,16,20,24,28

    vector<int> zone;                    // zone[cell]
    vector<vector<int>> lineCells;       // cells per line, in the order to print
    vector<ll> lineW;                    // weight per line

    auto newCell = [&](int z) -> int {
        zone.push_back(z);
        return (int)zone.size() - 1;
    };

    // ---- plain lines: fresh, mutually disjoint cells ----
    for (int i = 0; i < nPlain; i++) {
        int s = 2 + rnd.next(0, 2);                 // size 2..4
        vector<int> cells;
        for (int j = 0; j < s; j++) cells.push_back(newCell(rnd.next(0, 9)));
        ll w = 15 + rnd.next(0, 50);                 // 15..64
        lineCells.push_back(cells);
        lineW.push_back(w);
    }

    // ---- hub gadgets ----
    for (int g = 0; g < nGadget; g++) {
        int u = newCell(0);
        int v = newCell(5);
        int d = newCell(0);
        ll w2 = 150 + rnd.next(0, 100);              // hub-only line weight   150..249
        ll w1 = w2 + 40 + rnd.next(0, 60);            // decoy line weight      > w2, w1-w2 in [40,99]
        lineCells.push_back({u, v, d});
        lineW.push_back(w1);
        lineCells.push_back({u, v});
        lineW.push_back(w2);
    }

    int n = (int)zone.size();
    if (n % 2 == 1) { newCell(rnd.next(0, 9)); n++; }   // keep n even (unused padding cell)
    int m = (int)lineCells.size();

    // ---- budget: generous "afford every plain line" pool + exact gadget-hub funding ----
    auto pairCost = [&](int a, int b) -> ll { return 1 + (ll)abs(zone[a] - zone[b]); };
    ll K = 10;
    for (int i = 0; i < nPlain; i++) {
        auto &c = lineCells[i];
        ll best = LLONG_MAX;
        for (size_t a = 0; a < c.size(); a++)
            for (size_t b = a + 1; b < c.size(); b++)
                best = min(best, pairCost(c[a], c[b]));
        K += best;
    }
    K += (ll)nGadget * 6;

    // ---- threat weight per cell -> Aggressor's fixed claim order ----
    vector<ll> threat(n, 0);
    for (int i = 0; i < m; i++)
        for (int c : lineCells[i]) threat[c] += lineW[i];
    vector<int> order(n);
    for (int i = 0; i < n; i++) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) {
        if (threat[a] != threat[b]) return threat[a] > threat[b];
        return a < b;
    });

    // ---- emit ----
    printf("%d %d %lld\n", n, m, K);
    for (int i = 0; i < n; i++) printf("%d%c", zone[i], i + 1 == n ? '\n' : ' ');
    for (int i = 0; i < m; i++) {
        printf("%d %lld", (int)lineCells[i].size(), lineW[i]);
        for (int c : lineCells[i]) printf(" %d", c);
        printf("\n");
    }
    for (int i = 0; i < n; i++) printf("%d%c", order[i], i + 1 == n ? '\n' : ' ');
    return 0;
}
