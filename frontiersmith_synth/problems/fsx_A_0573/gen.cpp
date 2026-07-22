#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Cliff Amphitheater: Aiming the Echo".
// family: lattice-wave-focus-timing   objective: MAX
//
// A clapper S pulses at tick 0; M listeners want a loud momentary echo. We plant,
// per listener, a DISCRETE ISO-CHRONAL CLUSTER: cs cells that ALL share the exact
// integer arrival tick v under the octile lattice metric D=2*max+min, with matched
// facet parity so they add CONSTRUCTIVELY. Stacking a cluster gives peak ~ cs*g.
//
// TRAP (continuous optics): for each listener we also plant a EUCLIDEAN-ELLIPSE
// DECOY set -- cells sharing a constant *Euclidean* path sum (the textbook "focusing
// ring") but with SCATTERED integer arrival ticks and random parity. A solver that
// groups candidates by continuous distance (a parabolic/elliptical dish copied from
// optics) installs the decoys; on the lattice their true ticks smear across many
// bins so almost nothing stacks -> peak barely rises. The insight is to group by the
// TRUE integer arrival tick + parity (the iso-chronal cluster), which sits OFF the
// Euclidean ring. Decoys carry HIGHER gain to lure gain/continuous heuristics.
//
// Reflectors re-emit toward every listener, so an installed cell helps one listener
// and can land destructively in another's peak tick -> the M listeners are coupled,
// budget K is scarce, and no single-pass rule is optimal (open-ended allocation).
//
// Output:  G sx sy g0 M N K ; then M lines fx fy ; then N lines  x y g p .
// -----------------------------------------------------------------------------

static inline ll octile(ll ax, ll ay, ll bx, ll by){
    ll dx = llabs(ax-bx), dy = llabs(ay-by);
    ll mx = max(dx,dy), mn = min(dx,dy);
    return 2*mx + mn;
}
static inline double eucl(ll ax, ll ay, ll bx, ll by){
    double dx = (double)(ax-bx), dy = (double)(ay-by);
    return sqrt(dx*dx + dy*dy);
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    ll G  = 200 + (ll)llround(f * 1800.0);      // 200..2000
    ll sx = 8, sy = G/2;
    ll g0 = 10;
    int M = 3 + (int)llround(f * 17.0);          // 3..20
    int Ntarget = 40 + (int)llround(f * 2660.0); // 40..2700

    // trap intensity ladder: several testIds are pure-trap (decoys especially loud,
    // clusters modest) so continuous/gain heuristics land far from strong.
    bool hardTrap = (testId % 3 == 0) || testId >= 8;

    // ---- listeners on the far wall ----
    vector<ll> fx(M+1), fy(M+1);
    set<pair<ll,ll>> occupied;
    occupied.insert({sx,sy});
    for (int i = 1; i <= M; i++){
        ll yy = (M==1) ? G/2 : 10 + (ll)llround((double)(i-1)/(double)max(1,M-1) * (G-20));
        yy += rnd.next(-2, 2);
        yy = max(2LL, min(G-2, yy));
        ll xx = G - 8 + rnd.next(-2, 2);
        xx = max(2LL, min(G-1, xx));
        while (occupied.count({xx,yy})) { yy = max(2LL,min(G-2, yy + 1)); }
        fx[i] = xx; fy[i] = yy; occupied.insert({xx,yy});
    }

    struct Cell { ll x,y,g,p; };
    vector<Cell> cells;

    auto freshCell = [&](ll x, ll y)->bool{
        if (x < 1 || x > G || y < 1 || y > G) return false;
        if (occupied.count({x,y})) return false;
        return true;
    };

    ll K = 0;

    // ---- per-listener iso-chronal cluster + euclidean decoy ----
    int csBase = 4 + (int)llround(f * 2.0);       // 4..6
    for (int i = 1; i <= M; i++){
        int cs = csBase + rnd.next(0,1);           // cluster size (4..7)
        ll gm = hardTrap ? rnd.next(6,8) : rnd.next(7,9);   // cluster gain (modest in hard trap)

        // pick an achievable arrival tick v: sample a random off-axis cell, take its
        // arrival, require a moderate detour so cluster cells exist off the ring.
        ll v = -1;
        for (int att = 0; att < 4000 && v < 0; att++){
            ll x = rnd.next(20LL, G-20), y = rnd.next(2LL, G-2);
            if (!freshCell(x,y)) continue;
            ll a = octile(sx,sy,x,y) + octile(x,y,fx[i],fy[i]);
            ll base = octile(sx,sy,fx[i],fy[i]);
            if (a >= base + 6 && a <= base + 80) v = a;
        }
        if (v < 0) v = octile(sx,sy,fx[i],fy[i]) + 10;   // fallback (may yield few)

        ll pmatch = v & 1LL;                        // (v + pmatch) even -> constructive
        int placed = 0;
        for (int att = 0; att < 60000 && placed < cs; att++){
            ll x = rnd.next(20LL, G-20), y = rnd.next(2LL, G-2);
            if (!freshCell(x,y)) continue;
            ll a = octile(sx,sy,x,y) + octile(x,y,fx[i],fy[i]);
            if (a != v) continue;
            cells.push_back({x,y,gm,pmatch});
            occupied.insert({x,y});
            placed++;
        }
        K += placed;   // cluster cells are the "installable" budget backbone

        // euclidean-ellipse decoy: constant euclidean sum ~ ce, scattered true ticks,
        // HIGHER gain, random parity.
        int ds = cs + rnd.next(0,2);
        double baseE = eucl(sx,sy,fx[i],fy[i]);
        double ce = baseE * (1.25 + 0.15*rnd.next(0.0, 1.0));
        int dplaced = 0;
        for (int att = 0; att < 60000 && dplaced < ds; att++){
            ll x = rnd.next(20LL, G-20), y = rnd.next(2LL, G-2);
            if (!freshCell(x,y)) continue;
            double e = eucl(sx,sy,x,y) + eucl(x,y,fx[i],fy[i]);
            if (fabs(e - ce) > 2.0) continue;
            ll gh = hardTrap ? rnd.next(13,18) : rnd.next(10,14);   // loud lure
            cells.push_back({x,y,gh,(ll)rnd.next(0,1)});
            occupied.insert({x,y});
            dplaced++;
        }
    }

    // ---- noise fill to Ntarget ----
    int guard = 0;
    while ((int)cells.size() < Ntarget && guard < 400000){
        guard++;
        ll x = rnd.next(20LL, G-20), y = rnd.next(2LL, G-2);
        if (!freshCell(x,y)) continue;
        ll gl = rnd.next(3,7);
        cells.push_back({x,y,gl,(ll)rnd.next(0,1)});
        occupied.insert({x,y});
    }

    int N = (int)cells.size();
    if (N < 1){ // degenerate safety
        cells.push_back({G/2, G/3, 5, 0}); N = 1;
    }
    // budget: scarce relative to total cluster demand (forces allocation).
    K = max(2LL, (ll)llround(0.4 * (double)K));
    if (K > N) K = N;

    // shuffle candidate order (roles are not observable from position in list)
    for (int i = N-1; i > 0; i--) swap(cells[i], cells[rnd.next(0,i)]);

    printf("%lld %lld %lld %lld %d %d %lld\n", G, sx, sy, g0, M, N, K);
    for (int i = 1; i <= M; i++) printf("%lld %lld\n", fx[i], fy[i]);
    for (auto &c : cells) printf("%lld %lld %lld %lld\n", c.x, c.y, c.g, c.p);
    return 0;
}
