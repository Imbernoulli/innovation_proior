#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Factory heat-exchanger web"  (generator)  family: pinch-cascade-matching
//
// Streams that must be cooled (HOT) or heated (COLD) are matched by exchangers.
// Each exchanger moves heat from a hot stream to a cold stream; counter-current
// feasibility needs a minimum approach dTmin at BOTH ends. Objective is to
// MINIMIZE external utility (= total heat not recovered by exchangers), i.e. to
// MAXIMIZE recovered heat.
//
// PLANTED STRUCTURE (the "pinch"): the instance is built as TWO temperature
// clusters -- a LOW cluster and a HIGH cluster -- separated by a wide gap.
//   * Inside each cluster the hot streams sit dTmin+margin above the cold streams,
//     so every intra-cluster hot/cold pair is feasible and the cluster's heat is
//     (nearly) fully recoverable within itself.
//   * HIGH-cluster hots ARE hot enough to feed LOW-cluster colds (a huge driving
//     force), but LOW-cluster hots can NEVER reach HIGH-cluster colds.
// The gap between the clusters is the PINCH. Recovering heat requires matching
// WITHIN a cluster. The obvious greedy -- pair the hottest hot with the coldest
// cold -- transfers HIGH-cluster heat straight down to LOW-cluster colds, i.e.
// ACROSS the pinch. That strands the HIGH-cluster colds (only HIGH hots could
// serve them) and the LOW-cluster hots (their sinks are now taken), so it recovers
// roughly HALF of what a pinch-respecting design recovers.
//
// Cold heat modestly exceeds hot heat per cluster (surplus), so the maximum
// recovery is bounded by the hot side; full hot recovery needs SPLITTING a hot's
// duty across several colds. A simple one-to-one (no-split) pinch-respecting match
// leaves residuals -> head-room above the reference strong solution.
//
// Output:  NH NC dTmin
//          NH lines:  THs THt CP     (hot,  THs>THt)
//          NC lines:  TCs TCt CP     (cold, TCs<TCt)
// Streams are shuffled so a solver must INFER the clusters from temperatures.
// -----------------------------------------------------------------------------

struct Str{ ll s,t,cp; };

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int id = atoi(argv[1]);
    double f = (id - 1) / 9.0;

    const ll dTmin = 10, H = 100, M = H + 10;   // margin M>=H keeps intra-cluster feasible
    int nPer = 2 + (int)llround(f * 14.0);      // 2..16 streams per side per cluster
    // surplus of cold heat over hot heat, per cluster (bounds max recovery by hot side)
    double surplus = 0.12 + 0.06 * (rnd.next(0, 1000) / 1000.0);   // 0.12..0.18

    ll loBase = 100 + rnd.next(0, 40);
    ll hiBase = 600 + rnd.next(0, 60);

    vector<Str> hots, colds;

    auto build = [&](ll Cbot){
        ll hbot = Cbot + dTmin + M, htop = hbot + H;  // hot cluster [hbot,htop]
        vector<ll> cph(nPer);
        ll Sh = 0;
        for (int i = 0; i < nPer; i++){ cph[i] = 2 + rnd.next(0, 10); Sh += cph[i]; } // 2..12
        ll Sc = (ll)llround((double)Sh * (1.0 + surplus));
        if (Sc < nPer) Sc = nPer;
        vector<ll> cpc(nPer, 1);
        ll rem = Sc - nPer;
        for (ll k = 0; k < rem; k++) cpc[rnd.next(0, nPer - 1)]++;   // high-variance colds
        for (int i = 0; i < nPer; i++) hots.push_back({htop, hbot, cph[i]});
        for (int i = 0; i < nPer; i++) colds.push_back({Cbot, Cbot + H, cpc[i]});
    };
    build(loBase);
    build(hiBase);

    for (int i = (int)hots.size() - 1; i > 0; i--) swap(hots[i], hots[rnd.next(0, i)]);
    for (int i = (int)colds.size() - 1; i > 0; i--) swap(colds[i], colds[rnd.next(0, i)]);

    int NH = (int)hots.size(), NC = (int)colds.size();
    printf("%d %d %lld\n", NH, NC, dTmin);
    for (auto &h : hots)  printf("%lld %lld %lld\n", h.s, h.t, h.cp);
    for (auto &c : colds) printf("%lld %lld %lld\n", c.s, c.t, c.cp);
    return 0;
}
