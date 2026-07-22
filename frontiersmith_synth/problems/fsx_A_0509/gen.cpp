#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Sawmill with one dying thin blade"  (generator)  family: kerf-ledger-guillotine-planning
//
// A sheet W x H must be cut into a guillotine tree. Two blades:
//   thick kerf T (unlimited), thin kerf t<T (total thin-cut-length budget L).
// Each cut removes a kerf strip -> kerf loss depends on the cut TREE (topology).
//
// PLANTED STRUCTURE (the checker never sees these labels -- it re-derives shelves
// by grouping demands of equal height):
//   Every "shelf" is a horizontal strip holding a WIDE base piece (value 25, always
//   fits with the thick blade) plus a NARROW bonus piece that fits BESIDE the base
//   ONLY when both of that shelf's vertical cuts use the THIN blade (it saves
//   2*(T-t) of width, exactly enough). Upgrading a shelf costs 2*height of thin
//   cut-length from budget L; it yields the shelf's bonus value.
//
// So the whole instance is a knapsack: pick which shelves to spend thin budget on.
//   - EFFICIENT shelves: short (cheap upgrade), moderate bonus -> HIGH value/cost.
//   - PREMIUM  shelves: tall (expensive upgrade), slightly higher bonus -> LOW value/cost.
// Budget L == total cost of all efficient shelves.  The obvious greedy (grab the
// highest-VALUE bonus first) buys premiums, blows the budget, and starves the many
// efficient shelves. The insight is to ration by VALUE PER UNIT BUDGET (density) --
// "marginal kerf saved per piece" -- which fills up on efficient shelves for far
// more total value within the same L.
//
// Heights are globally distinct (efficient block below a premium block) so a solver
// can reconstruct shelves by grouping same-height demands.
//
// Output:  W H T t L D   then D lines  w h v .
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    ll T = 4 + (testId % 3);          // 4..6  thick kerf
    ll t = 1;                          // thin kerf
    ll Sg = 2*T + 6;                   // per-shelf non-base width slack
    ll W = 200 + (ll)llround(f * 5800.0);   // 200..6000
    ll WB = W - Sg;                    // wide base width (>=1, huge)

    int nEff  = 8  + (int)llround(f * 80.0);   // 8..88  efficient shelves
    int nPrem = 12 + (int)llround(f * 48.0);   // 12..60 premium shelves
    ll HE0 = 8;                                 // efficient heights: HE0+1 .. HE0+nEff
    ll HP0 = HE0 + nEff + 50;                    // premium heights strictly above
    ll baseVal = 25;

    struct Dem { ll w, h, v; };
    vector<Dem> dem;
    ll totalEffCost = 0;               // == budget L

    // ---- efficient shelves ----
    for (int g = 1; g <= nEff; g++){
        ll h = HE0 + g;
        ll vE = 280 + rnd.next(0, 40);              // 280..320
        ll WN = 7 + rnd.next(0, (int)(2*(T - t) - 2)); // narrow bonus width in [7, 2T+3]
        dem.push_back({WB, h, baseVal});
        dem.push_back({WN, h, vE});
        totalEffCost += 2 * h;
    }
    // ---- premium shelves (higher value, much higher cost -> lower density) ----
    for (int j = 1; j <= nPrem; j++){
        ll h = HP0 + j;
        ll vP = 340 + rnd.next(0, 60);              // 340..400  (> any efficient value)
        ll WN = 7 + rnd.next(0, (int)(2*(T - t) - 2));
        dem.push_back({WB, h, baseVal});
        dem.push_back({WN, h, vP});
    }

    int G = nEff + nPrem;
    ll L = totalEffCost;               // budget: exactly the efficient set's cost

    // ---- sheet height: stack all shelves with thick horizontal cuts + final waste ----
    ll sumh = 0;
    for (int g = 1; g <= nEff;  g++) sumh += HE0 + g;
    for (int j = 1; j <= nPrem; j++) sumh += HP0 + j;
    ll H = sumh + (ll)G * T + (ll)(1 + rnd.next(0, 20));

    // shuffle demand listing order (grouping by height still recovers shelves)
    for (int i = (int)dem.size() - 1; i > 0; i--) swap(dem[i], dem[rnd.next(0, i)]);

    int D = (int)dem.size();
    printf("%lld %lld %lld %lld %lld %d\n", W, H, T, t, L, D);
    for (auto &d : dem) printf("%lld %lld %lld\n", d.w, d.h, d.v);
    return 0;
}
