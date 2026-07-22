#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Laser cutter that warps its own sheet"  (generator)
// family: pierce-heat-choreography
//
// A square sheet (N x N cells) holds axis-aligned rectangular PARTS. The solver
// chooses WHICH parts to place, WHERE (non-overlapping, in bounds), and in what
// ORDER to cut them (the k-th line cut at contour-tick k). Cutting a part deposits
// its pierce heat q; that heat then decays with elapsed ticks and attenuates with
// cell separation. A part scraps if the field at it, at the moment IT is cut,
// exceeds THR. Only surviving parts score their value v; a scrapped part still
// deposited its heat.
//
// The three composed mechanisms:
//   * thermal-decay-field   -- transient heat felt = q_i / (1 + DA*dt + DB*gap)
//                              (reciprocal decay in BOTH time and space), PLUS a
//                              time-INDEPENDENT persistent warp max(0, PW - PG*gap)
//                              a nearby cut leaves in the metal forever.
//   * irregular-nesting     -- heterogeneous rectangles; packing pressure (total
//                              part area ~ 0.3-0.45 of the sheet) forces proximity.
//   * contour-order-coupling-- only parts cut EARLIER heat a part; so the schedule
//                              (order of the single placement list) sets every dt.
//
// The obvious approach an average strong coder writes: cram parts densely to
// maximize placed value, then be clever about the cut ORDER (a cooling schedule).
// TRAP (planted every test): the PERSISTENT warp max(0, PW - PG*gap) a nearby cut
// leaves is TIME-INDEPENDENT, so no cut order can remove it. At gap 0 every touching
// neighbour deposits PW; in a max-density pack each part has several touching
// neighbours, so their warp alone exceeds THR -- max-density nesting is thermally
// unsalvageable by ANY schedule. The insight: the true variable is a SPACE-TIME
// layout -- leave thermal gaps (gap >= PW/PG kills the warp) AND schedule the
// remaining transient adjacencies apart, and CUT ONLY the parts that will survive
// (a doomed cut still heats its neighbours). Co-design placement + schedule + subset.
//
// Output:   N M DA DB PW PG THR
//           then M lines:  w h v q
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // part count grows with testId
    ll M = (ll)llround((0.08 + 0.92 * f) * 420.0);  // ~34 .. 420
    M = max<ll>(8, min<ll>(420, M));
    if (testId == 1) M = 12;

    // SMALL parts (dims 1..s). Small parts are what make a warp-clearing gap a REAL
    // density sacrifice (the pitch roughly doubles), so co-design is a genuine
    // trade-off rather than a free lunch.
    ll s = 4;
    // sheet sized for packing pressure: a dense grid (cell ~ s) fits about M parts,
    // so a solver that leaves thermal gaps must give up a large fraction of them.
    ll N = (ll)llround(1.15 * (double)s * sqrt((double)M));
    N = max<ll>(24, min<ll>(500, N));

    // thermal weights. transient q/(1+DA*dt+DB*gap) cools with ticks AND space;
    // persistent warp max(0,PW-PG*gap) is TIME-INDEPENDENT and only spacing removes
    // it (warp reaches 0 at gap >= PW/PG cells -- a big gap for these small parts).
    ll DA = 1, DB = 2;
    ll PW = 100, PG = 25;                           // warp zero at gap >= 4 cells
    bool tightTrap = (testId == 4 || testId == 7 || testId == 9 || testId == 10);
    ll THR = tightTrap ? 340 : 390;

    // ---- build the parts ----
    struct Dem { ll w, h, v, q; };
    vector<Dem> dem;
    // planted "hot & valuable" parts that MUST be spaced/isolated (any touching
    // neighbour's warp scraps them); the obvious dense pack destroys them.
    int nHot = 1 + (int)llround(f * 6.0);          // 1 .. 7
    if (testId == 4) nHot = 1;                      // NEEDLE: a single huge-value part
    for (int i = 0; i < M; i++){
        ll w = rnd.next((ll)1, s);
        ll h = rnd.next((ll)1, s);
        ll v = rnd.next((ll)80, (ll)120);
        ll q = rnd.next((ll)220, (ll)270);          // tight band: 2 touching -> scrap
        dem.push_back({w, h, v, q});
    }
    for (int k = 0; k < nHot && k < (int)dem.size(); k++){
        int idx = rnd.next(0, (int)dem.size() - 1);
        ll v = (testId == 4 && k == 0) ? 2600 : rnd.next((ll)260, (ll)440);
        ll q = rnd.next((ll)270, (ll)300);
        dem[idx].v = v;
        dem[idx].q = q;
    }

    // ---- emit ----
    printf("%lld %lld %lld %lld %lld %lld %lld\n", N, M, DA, DB, PW, PG, THR);
    for (int i = 0; i < (int)dem.size(); i++)
        printf("%lld %lld %lld %lld\n", dem[i].w, dem[i].h, dem[i].v, dem[i].q);
    return 0;
}
