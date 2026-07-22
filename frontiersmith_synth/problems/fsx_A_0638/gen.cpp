#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Generator for "Staggered Leaky Windbreaks Against the Steppe Gale".
// Emits:
//   W H Wb Pmax
//   Ymin Ymax
//   K
//   inflow[0..H-1]
//   driftNum driftDen
//   cost[1..Pmax]
//   drag[1..Pmax]
//   jetfrac[1..Pmax]
//   M
//   M lines: x y

static const ll PMAX = 4;

struct Tables { vector<ll> cost, drag, jet; };

static Tables genTables(){
    Tables t; t.cost.assign(PMAX + 1, 0); t.drag.assign(PMAX + 1, 0); t.jet.assign(PMAX + 1, 0);
    t.cost[1] = rnd.next(1, 3);
    t.cost[2] = t.cost[1] + rnd.next(2, 4);
    t.cost[3] = t.cost[2] + rnd.next(3, 6);
    t.cost[4] = t.cost[3] + rnd.next(4, 8);
    t.drag[1] = rnd.next(150, 250);
    t.drag[2] = t.drag[1] + rnd.next(100, 200);
    t.drag[3] = t.drag[2] + rnd.next(100, 200);
    t.drag[4] = min((ll)950, t.drag[3] + rnd.next(80, 150));
    t.jet[1] = rnd.next(50, 150);
    t.jet[2] = t.jet[1] + rnd.next(100, 200);
    t.jet[3] = t.jet[2] + rnd.next(150, 250);
    t.jet[4] = min((ll)950, t.jet[3] + rnd.next(100, 200));
    return t;
}

// A deliberately sharp-contrast table for the planted trap cases: level 1 is very
// cheap, low-drag and almost pure diffuse (leaky); level Pmax is expensive, removes
// most of the local speed, and is almost pure jet (solid) -- so the leaky-vs-solid
// tension the checker's mechanics create is unambiguous on these tests.
static Tables sharpTables(){
    Tables t; t.cost = {0,1,3,8,20}; t.drag = {0,280,520,700,850}; t.jet = {0,60,280,550,780};
    return t;
}

static void printHeader(ll W, ll H, ll Wb, ll Ymin, ll Ymax, ll K,
                         const vector<ll>& inflow, ll driftNum, ll driftDen, const Tables& t){
    printf("%lld %lld %lld %lld\n", W, H, Wb, PMAX);
    printf("%lld %lld\n", Ymin, Ymax);
    printf("%lld\n", K);
    for (ll y = 0; y < H; y++) printf("%lld%c", inflow[y], y + 1 == H ? '\n' : ' ');
    printf("%lld %lld\n", driftNum, driftDen);
    for (ll p = 1; p <= PMAX; p++) printf("%lld%c", t.cost[p], p == PMAX ? '\n' : ' ');
    for (ll p = 1; p <= PMAX; p++) printf("%lld%c", t.drag[p], p == PMAX ? '\n' : ' ');
    for (ll p = 1; p <= PMAX; p++) printf("%lld%c", t.jet[p], p == PMAX ? '\n' : ' ');
}

static void printCrops(const vector<pair<ll,ll>>& crops){
    printf("%d\n", (int)crops.size());
    for (auto& c : crops) printf("%lld %lld\n", c.first, c.second);
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    if (testId == 1){
        // tiny sanity (statement-example scale)
        ll W = 6, H = 4, Wb = 2, Ymin = 1, Ymax = 2, K = 12;
        vector<ll> inflow = {100, 300, 280, 120};
        ll driftNum = 0, driftDen = 1;
        Tables t; t.cost = {0,1,2,4,7}; t.drag = {0,200,400,650,850}; t.jet = {0,100,300,600,900};
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        vector<pair<ll,ll>> crops = {{2,1},{3,2},{4,0}};
        printCrops(crops);
    } else if (testId == 2 || testId == 4 || testId == 6 || testId == 9){
        // generic random, growing scale
        ll W, H, Wb, M, Ymin, Ymax;
        if (testId == 2){ W = 16; H = 8; Wb = 3; Ymin = 2; Ymax = 4; M = rnd.next(8,15); }
        else if (testId == 4){ W = 34; H = 16; Wb = 13; Ymin = 5; Ymax = 9; M = rnd.next(30,50); }
        else if (testId == 6){ W = 60; H = 26; Wb = 22; Ymin = 8; Ymax = 14; M = rnd.next(80,140); }
        else { W = 100; H = 60; Wb = 30; Ymin = 20; Ymax = 29; M = rnd.next(150,260); }
        vector<ll> inflow(H);
        for (ll y = 0; y < H; y++) inflow[y] = rnd.next(80, 900);
        ll driftDen = rnd.next(5, 25);
        ll driftNum = rnd.next(0, (int)driftDen - 1);
        Tables t = sharpTables();
        // budget is always at least enough for full leaky (level-1) coverage of the
        // whole zone, with room to spare -- so the ladder's spread reflects placement
        // skill, not an accidental budget shortfall.
        ll bandCost1Full = t.cost[1] * Wb * (Ymax - Ymin + 1);
        ll K = rnd.next((int)bandCost1Full, (int)(bandCost1Full * 3 + 5));
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        // crops stay within a bounded near-field window downwind of the windbreak
        // zone: far enough downwind that the drift/diffuse mechanics matter, but not
        // so far that repeated rotation/diffusion washes the treatment out entirely.
        // Lanes stay close to the plantable band (a couple lanes of slack for tip
        // effects) -- an orchard planted far outside anything the windbreak could
        // ever reach wouldn't be an interesting design choice.
        ll cropXHi = min(W - 1, Wb + 20);
        ll capacity = (cropXHi - Wb + 1) * (Ymax - Ymin + 1);
        M = min(M, max((ll)1, capacity - capacity / 10));
        // crops sit strictly inside the plantable band: a windbreak this deep (Wb
        // columns) genuinely helps them the more of its depth it uses, so the ladder
        // reflects real placement/depth skill (dedicated trap tests 3, 5, 7, 10
        // separately cover the tip-jet / drift-misalignment / spreading mechanics).
        vector<pair<ll,ll>> crops;
        set<pair<ll,ll>> seen;
        while ((ll)crops.size() < M){
            ll x = rnd.next((int)Wb, (int)cropXHi);
            ll y = rnd.next((int)Ymin, (int)Ymax);
            if (seen.insert({x,y}).second) crops.push_back({x,y});
        }
        printCrops(crops);
    } else if (testId == 3){
        // PLANTED trap: tip jets. A band narrower than H; generous K so a fully
        // solid single-row wall across the whole band is easily affordable.
        // Crops sit both INSIDE the shadow (reward real shielding) and right at
        // the band's two tips (punished hard by a solid wall's edge jets).
        ll W = 14, H = 24, Wb = 4;
        ll Ymin = 8, Ymax = 15;
        vector<ll> inflow(H);
        for (ll y = 0; y < H; y++) inflow[y] = 380 + (y % 5) * 30;
        ll driftNum = 0, driftDen = 1;
        Tables t = sharpTables();
        ll K = t.cost[4] * (Ymax - Ymin + 1) + t.cost[2] * 3; // solid single row, plus slack
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        vector<pair<ll,ll>> crops;
        for (ll x = Wb; x < W; x++){
            crops.push_back({x, Ymin - 1});
            crops.push_back({x, Ymax + 1});
        }
        for (ll x = Wb; x < W; x += 2){
            crops.push_back({x, (Ymin+Ymax)/2});
            crops.push_back({x, (Ymin+Ymax)/2 + 1});
        }
        printCrops(crops);
    } else if (testId == 5){
        // PLANTED trap: drift misalignment + misdirected loudness. A handful of
        // windbreak lanes are LOUD (high inflow) but shield nothing downwind; one
        // QUIET lane's relief, carried by the crosswind drift, is exactly what
        // reaches a deep crop cluster many columns later.
        ll W = 70, H = 20, Wb = 5;
        ll Ymin = 3, Ymax = 16;
        ll driftDen = 9, driftNum = 2; // ~0.222 lane/column
        vector<ll> inflow(H, 120);
        vector<ll> loud = {4, 7, 12}; // decoys: high inflow, nothing downwind
        for (ll y : loud) inflow[y] = 900;
        ll sourceLane = 9; // quiet on purpose
        inflow[sourceLane] = 90;
        Tables t = sharpTables();
        ll K = t.cost[3] * 3; // only enough to seriously treat ~2-3 lanes well
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        // crop cluster deep downwind, at the lane the source-lane shadow drifts into
        ll deepX0 = W - 6;
        vector<pair<ll,ll>> crops;
        for (ll dx = 0; dx < 6; dx++){
            ll x = deepX0 + dx;
            ll shiftLanes = (x * driftNum / driftDen) % H;
            ll targetLane = ((sourceLane + shiftLanes) % H + H) % H;
            crops.push_back({x, targetLane});
            if (targetLane + 1 < H) crops.push_back({x, targetLane + 1});
        }
        // a few decoy crops far from anything loud protects, to keep it non-degenerate
        for (ll x = Wb; x < Wb + 4; x++) crops.push_back({x, 1});
        printCrops(crops);
    } else if (testId == 7){
        // PLANTED trap: convexity / spreading. Crops spread evenly across the full
        // plantable band; budget can EITHER solidly wall half the band OR leakily
        // treat the whole band. Sum-of-squares over evenly spread crops rewards
        // spreading (thin, even relief) over concentration (half exposed at full
        // speed, plus a jet at the covered/uncovered seam).
        ll W = 30, H = 26, Wb = 6;
        ll Ymin = 2, Ymax = 23; // wide band, 22 lanes
        vector<ll> inflow(H);
        for (ll y = 0; y < H; y++) inflow[y] = 500 + (y % 3) * 40;
        ll driftNum = 0, driftDen = 1;
        Tables t = sharpTables();
        ll bandW = Ymax - Ymin + 1;
        ll K = t.cost[4] * (bandW/2) + t.cost[1] * 2; // ~ half band solid, tuned for the tradeoff
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        vector<pair<ll,ll>> crops;
        for (ll x = Wb; x < W; x += 2){
            for (ll y = Ymin; y <= Ymax; y++) crops.push_back({x, y});
        }
        printCrops(crops);
    } else if (testId == 8){
        // NEEDLE: one very loud lane hidden among many quiet ones; crops cluster
        // right behind it (plus decoys elsewhere).
        ll W = 26, H = 22, Wb = 5;
        ll Ymin = 0, Ymax = H - 1;
        vector<ll> inflow(H);
        for (ll y = 0; y < H; y++) inflow[y] = rnd.next(60, 140);
        ll needle = rnd.next(0, (int)H - 1);
        inflow[needle] = 1800;
        ll driftDen = 11, driftNum = 1;
        Tables t = genTables();
        ll K = t.cost[4] * 3;
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        vector<pair<ll,ll>> crops;
        for (ll x = Wb; x < W; x++){
            ll shiftLanes = (x * driftNum / driftDen) % H;
            ll targetLane = ((needle + shiftLanes) % H + H) % H;
            crops.push_back({x, targetLane});
        }
        for (int i = 0; i < 12; i++){
            ll x = rnd.next((int)Wb, (int)W - 1);
            ll y = rnd.next(0, (int)H - 1);
            crops.push_back({x, y});
        }
        printCrops(crops);
    } else { // testId == 10 : large adversarial, combines traps at max scale
        ll W = 100, H = 60, Wb = 30;
        ll Ymin = 24, Ymax = 33; // narrower than H -> tip jets possible, narrow enough for Wb to work it
        vector<ll> inflow(H);
        for (ll y = 0; y < H; y++) inflow[y] = rnd.next(100, 950);
        ll driftDen = rnd.next(6, 15);
        ll driftNum = rnd.next(1, (int)driftDen - 1);
        Tables t = sharpTables();
        ll bandW = Ymax - Ymin + 1;
        ll K = t.cost[1] * Wb * bandW * 2; // comfortably affords full leaky coverage twice over
        printHeader(W,H,Wb,Ymin,Ymax,K,inflow,driftNum,driftDen,t);
        ll cropXHi = min(W - 1, Wb + 20);
        vector<pair<ll,ll>> crops;
        set<pair<ll,ll>> seen;
        // a MINORITY of crops sit right at the tips (punished by a solid wall's edge
        // jets); most sit inside the band (a solid wall genuinely helps them).
        for (ll x = Wb; x < cropXHi; x += 5){
            for (ll dy : {-1LL,1LL}){
                ll y = Ymin + dy; if (y < 0) continue; if (y >= H) continue;
                if (seen.insert({x,y}).second) crops.push_back({x,y});
                y = Ymax + dy;
                if (y >= 0 && y < H && seen.insert({x,y}).second) crops.push_back({x,y});
            }
        }
        // dense random fill to the constraint envelope, strictly inside the band
        // (still near-field: W itself stays at the size cap to stress-test
        // generator/checker performance, but crop signal is kept where the
        // windbreak can still plausibly matter)
        ll capacity = (cropXHi - Wb + 1) * (Ymax - Ymin + 1);
        ll target = min((ll)700, capacity - capacity / 10);
        while ((ll)crops.size() < target){
            ll x = rnd.next((int)Wb, (int)cropXHi);
            ll y = rnd.next((int)Ymin, (int)Ymax);
            if (seen.insert({x,y}).second) crops.push_back({x,y});
        }
        printCrops(crops);
    }
    return 0;
}
