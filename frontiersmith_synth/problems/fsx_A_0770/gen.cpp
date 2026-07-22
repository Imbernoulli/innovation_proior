#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Silencing a War-Drum Rehearsal Hall"   family: room-mode-absorber-tuning
//
// Physical picture: a rectangular hall W x H with reflecting walls. A war-drum
// at (sx,sy) pulses periodically with period P; the room's response is modeled
// as a sum of standing-wave "resonance channels" (m,n), 0<=m,n<=MMAX, plus one
// extra pseudo-channel that stands in for the direct + first-order specular
// reflection path (computed geometrically via the mirror-image construction).
//
// Each channel k has: Delta_k (how far the drum's drive frequency is from that
// channel's natural resonance -- small Delta = dangerously resonant), Gamma0_k
// (baseline wall damping, fixed), S_k (how strongly the drum excites it, i.e.
// the channel's standing-wave amplitude AT THE SOURCE), L_k (how strongly the
// channel is heard at the listeners, i.e. its amplitude AT THE LISTENERS,
// summed in energy over multiple listeners).
//
// A wall panel candidate at (x,y) coupled into channel k by phi2 = (channel
// k's standing-wave amplitude AT (x,y))^2 -- this is the mechanism: damping
// effectiveness at a panel is proportional to the modal amplitude AT THAT
// PANEL, not at the listener. Panels are all placed on the wall ring; cost and
// absorption strength (alpha) are independent random draws, NOT correlated
// with position, so proximity to the listener reveals nothing about a panel's
// true value -- only the phi2 table (which encodes real room-mode geometry)
// does.
//
// PLANTED TRAP (>=3 of 10 tests): drum near a room corner, listener(s) at/near
// the room center. The center is an exact node of every mode with odd m or odd
// n (cos(m*pi*0.5)=0), while a corner is an antinode of EVERY mode. The drive
// period P is tuned near a genuine low-order resonance (m,n)=(1,1) or similar,
// so that resonant channel's S is huge (drum sits at its antinode) while
// treating panels near the listener barely touches that channel's Gamma
// (phi2 near-listener is near 0 there) -- you must treat panels near the
// SOURCE / at the mode's antinodes, which are far from the listener, to kill
// the dominant channel.
// -----------------------------------------------------------------------------

static const ll FX          = 1000000LL;   // phi2 / L scale
static const ll SRC_AMP     = 100000LL;    // S scale
static const ll DELTA_SCALE = 1000000LL;   // Delta scale
static const ll GAMMA0_CONST = 6000000LL;  // flat baseline damping per channel
static const double KAPPA = 0.2002;        // lattice wave-speed^2 (only used here, for omega)

struct Cand { int x, y; };

vector<Cand> buildPerimeter(int W, int H){
    vector<Cand> v;
    for (int x = 0; x <= W - 1; x++) v.push_back({x, 0});                 // top row
    for (int y = 1; y <= H - 2; y++) v.push_back({W - 1, y});             // right col
    for (int x = W - 2; x >= 0; x--) v.push_back({x, H - 1});             // bottom row
    for (int y = H - 2; y >= 1; y--) v.push_back({0, y});                 // left col
    return v;
}

double phiAxis(int m, int coord, int dim){
    if (m == 0) return 1.0;
    return cos(M_PI * m * (double)coord / (double)(dim - 1));
}
double phi(int m, int n, int x, int y, int W, int H){
    return phiAxis(m, x, W) * phiAxis(n, y, H);
}

// reflect `src` across the vertical line x=wallX, intersect the segment to `lst`
// with that line; return the y-coordinate of the intersection (or -1 if degenerate).
double reflectVert(double wallX, double sx, double sy, double lx, double ly){
    double sxp = 2 * wallX - sx;
    double denom = (lx - sxp);
    if (fabs(denom) < 1e-9) return -1;
    double t = (wallX - sxp) / denom;
    return sy + t * (ly - sy);
}
double reflectHorz(double wallY, double sx, double sy, double lx, double ly){
    double syp = 2 * wallY - sy;
    double denom = (ly - syp);
    if (fabs(denom) < 1e-9) return -1;
    double t = (wallY - syp) / denom;
    return sx + t * (lx - sx);
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int W, H, sx, sy, MT, NT; // room, source, target resonant mode (m,n)
    vector<pair<int,int>> listeners;
    bool trap;

    switch (testId){
        // trap cases: source near a corner (antinode of every low mode), listener(s)
        // kept near the room CENTER (an exact node of every odd-(m or n) mode).
        case 1: W=10; H=10; sx=1; sy=1; listeners={{4,4}}; MT=1; NT=1; trap=true; break;
        case 2: W=14; H=12; sx=6; sy=4; listeners={{2,9}}; MT=2; NT=1; trap=false; break;
        case 3: W=16; H=18; sx=9; sy=3; listeners={{4,10},{12,14}}; MT=1; NT=3; trap=false; break;
        case 4: W=20; H=20; sx=2; sy=2; listeners={{9,9}}; MT=1; NT=1; trap=true; break;
        case 5: W=22; H=18; sx=2; sy=2; listeners={{10,8}}; MT=1; NT=1; trap=true; break;
        case 6: W=24; H=26; sx=2; sy=2; listeners={{11,12}}; MT=1; NT=2; trap=true; break;
        case 7: W=20; H=20; sx=2; sy=17; listeners={{9,9},{10,11}}; MT=1; NT=1; trap=true; break;
        case 8: W=28; H=28; sx=2; sy=2; listeners={{13,13}}; MT=1; NT=1; trap=true; break;
        case 9: W=30; H=24; sx=27; sy=2; listeners={{14,11},{15,12}}; MT=2; NT=1; trap=true; break;
        case 10: W=36; H=32; sx=2; sy=2; listeners={{17,15},{18,16}}; MT=1; NT=1; trap=true; break;
        default: W=20; H=20; sx=2; sy=2; listeners={{9,9}}; MT=1; NT=1; trap=true; break;
    }
    (void)trap;
    int NL = (int)listeners.size();

    // ---- pick drive period P near the target mode's resonance ----
    double kx = M_PI * MT / (double)(W - 1);
    double ky = M_PI * NT / (double)(H - 1);
    double omega = sqrt(KAPPA) * sqrt(kx*kx + ky*ky);
    int P = max(3, (int)llround(2 * M_PI / omega));
    // small deterministic jitter (still near-resonant) so tests aren't identical patterns
    P += (testId % 3) - 1;
    if (P < 3) P = 3;
    double omegaDrive = 2 * M_PI / P;

    // ---- mode table: (m,n) in [0..MMAX]x[0..NMAX], plus 1 first-reflection channel ----
    int MMAX = 5, NMAX = 5;
    vector<array<int,2>> modes;
    for (int m = 0; m <= MMAX; m++)
        for (int n = 0; n <= NMAX; n++)
            modes.push_back({m, n});
    int NMreal = (int)modes.size();
    int NM = NMreal + 1; // + first-reflection pseudo-channel (last index)

    vector<ll> Delta(NM), Gamma0(NM, GAMMA0_CONST), Sc(NM), Lc(NM, 0);
    for (int k = 0; k < NMreal; k++){
        int m = modes[k][0], n = modes[k][1];
        double kxx = M_PI * m / (double)(W - 1);
        double kyy = M_PI * n / (double)(H - 1);
        double om = sqrt(KAPPA) * sqrt(kxx*kxx + kyy*kyy);
        Delta[k] = (ll)llround(DELTA_SCALE * fabs(om - omegaDrive));
        double phiSrc = phi(m, n, sx, sy, W, H);
        Sc[k] = (ll)llround(SRC_AMP * fabs(phiSrc));
        double sumL2 = 0;
        for (auto& l : listeners){
            double phL = phi(m, n, l.first, l.second, W, H);
            sumL2 += phL * phL;
        }
        Lc[k] = (ll)llround(FX * sumL2);
        if (Lc[k] > 5000000) Lc[k] = 5000000;
    }
    // first-reflection pseudo-channel: no detuning, drum always couples fully,
    // heard fully at the listeners -- but only the geometrically-correct panels
    // can damp it (handled via phi2 below).
    Delta[NMreal] = 0;
    Sc[NMreal] = SRC_AMP;
    Lc[NMreal] = (ll)min((ll)5000000, (ll)NL * FX);

    // ---- candidates: full perimeter wall ring ----
    vector<Cand> ring = buildPerimeter(W, H);
    int M = (int)ring.size();

    // map coord -> candidate index (1-based) for the first-reflection flag lookup
    map<pair<int,int>,int> coordToId;
    for (int i = 0; i < M; i++) coordToId[{ring[i].x, ring[i].y}] = i + 1;

    auto snapToRing = [&](double cx, double cy, int wallAxis, double wallCoordFixed)->pair<int,int>{
        // wallAxis==0: vertical wall at x=wallCoordFixed, cy is the free coordinate
        // wallAxis==1: horizontal wall at y=wallCoordFixed, cx is the free coordinate
        if (wallAxis == 0){
            int yy = (int)llround(cy);
            yy = max(1, min(H - 2, yy));
            return {(int)llround(wallCoordFixed), yy};
        } else {
            int xx = (int)llround(cx);
            xx = max(1, min(W - 2, xx));
            return {xx, (int)llround(wallCoordFixed)};
        }
    };

    set<int> frFlagged;
    for (auto& l : listeners){
        double lx = l.first, ly = l.second;
        // left wall x=0
        { double yy = reflectVert(0, sx, sy, lx, ly);
          if (yy > -0.5){ auto c = snapToRing(0, yy, 0, 0); auto it = coordToId.find(c); if (it != coordToId.end()) frFlagged.insert(it->second); } }
        // right wall x=W-1
        { double yy = reflectVert(W - 1, sx, sy, lx, ly);
          if (yy > -0.5){ auto c = snapToRing(0, yy, 0, W - 1); auto it = coordToId.find(c); if (it != coordToId.end()) frFlagged.insert(it->second); } }
        // top wall y=0
        { double xx = reflectHorz(0, sx, sy, lx, ly);
          if (xx > -0.5){ auto c = snapToRing(xx, 0, 1, 0); auto it = coordToId.find(c); if (it != coordToId.end()) frFlagged.insert(it->second); } }
        // bottom wall y=H-1
        { double xx = reflectHorz(H - 1, sx, sy, lx, ly);
          if (xx > -0.5){ auto c = snapToRing(xx, 0, 1, H - 1); auto it = coordToId.find(c); if (it != coordToId.end()) frFlagged.insert(it->second); } }
    }

    // ---- per-candidate cost / alpha (independent of position) + phi2 table ----
    vector<ll> cost(M + 1), alpha(M + 1);
    vector<vector<ll>> phi2(M + 1, vector<ll>(NM));
    ll totalCost = 0;
    for (int i = 1; i <= M; i++){
        cost[i] = rnd.next(1, 10);
        alpha[i] = rnd.next(200, 900);
        totalCost += cost[i];
        int cx = ring[i - 1].x, cy = ring[i - 1].y;
        for (int k = 0; k < NMreal; k++){
            int m = modes[k][0], n = modes[k][1];
            double ph = phi(m, n, cx, cy, W, H);
            phi2[i][k] = (ll)llround(FX * ph * ph);
        }
        phi2[i][NMreal] = frFlagged.count(i) ? FX : 0;
    }

    ll Budget = max((ll)5, (ll)llround(0.11 * (double)totalCost));

    // ---- emit ----
    printf("%d %d\n", W, H);
    printf("%d %d\n", sx, sy);
    printf("%d\n", NL);
    for (auto& l : listeners) printf("%d %d\n", l.first, l.second);
    printf("%d\n", NM);
    for (int k = 0; k < NM; k++) printf("%lld %lld %lld %lld\n", Delta[k], Gamma0[k], Sc[k], Lc[k]);
    printf("%lld\n", Budget);
    printf("%d\n", M);
    for (int i = 1; i <= M; i++){
        printf("%d %d %d %lld %lld", i, ring[i-1].x, ring[i-1].y, cost[i], alpha[i]);
        for (int k = 0; k < NM; k++) printf(" %lld", phi2[i][k]);
        printf("\n");
    }
    return 0;
}
