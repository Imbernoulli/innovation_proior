#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Stencil Counters: Bridging the Floating Middles".
// family: bridge-anchored-stencil
//
// Layout: an SxS grid whose outer border ring is the FRAME ('#'). Up to 3 solid
// square islands ("counters") are placed at distinct CORNERS (top-left,
// top-right, bottom-left) of the interior, each with the SAME gap length L on
// its two "home" sides (the two borders it faces directly) -- e.g. the top-left
// island is L cells from the top border and L cells from the left border. Home
// sides are always unobstructed by any other island (no other island shares
// both its row range and its column range), so the geometrically nearest
// direction is always genuinely buildable. The far sides may be blocked by
// another island or simply far away -- irrelevant, since no solver benefits
// from choosing them (they are never the minimum-gap direction).
//
// PLANTED TRAP: every island has the SAME gap length L on both of its home
// sides. A single rod bridging it via ONE home side must supply BOTH the
// (cheap, axial) stiffness along that side's axis AND the (expensive,
// cubic-bending) stiffness across it -- so its width is dictated by the weak
// bending term, growing like L * Sreq^(1/3) and costing ~ L^2 * Sreq^(1/3).
// Two orthogonal rods (one on each home side) can each satisfy their OWN axis
// via the cheap axial term alone, costing ~ 2 * L^2 * Sreq / A -- with A tuned
// generously large and BD modest, this is far cheaper. Islands are sized to
// just barely allow the expensive single-rod option (so it is legal, merely
// wasteful), and the visual-cost map is skewed (cost 3 vs 1) near every
// structure, further penalizing the wide single rod's larger footprint.
// -----------------------------------------------------------------------------

const int A = 22, BD = 4;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // NOTE: the single-vs-pair cost RATIO depends only on Sreq (both costs scale
    // as L^2, so L cancels) -- keep Sreq in a fixed safe band across all tests
    // (tuned so strong's score stays well under the 0.92 headroom cap) and use L,
    // K purely to grow the difficulty/size ladder.
    ll L = 15 + (ll)llround(f * 20.0);             // 15..35
    ll SreqBase = 7;                               // +jitter 0..2 below
    int K = (testId <= 4) ? 1 : (testId <= 8 ? 2 : 3);
    const ll margin = 3;

    ll Leff = margin + L; // actual home-side gap length every island will see
    vector<ll> Sreq(K + 1);
    ll IS = 6;
    for (int i = 1; i <= K; i++) {
        Sreq[i] = SreqBase + rnd.next(0, 2);
        double wb = ceil((double)Leff * cbrt((double)Sreq[i] * BD) - 1e-9);
        ll need = (ll)ceil(1.4 * wb - 1e-9) + 6;
        IS = max(IS, need);
    }

    ll gapMid = 4 * L + 10;
    ll S = 2 * margin + 2 * L + 2 * IS + gapMid;
    ll R = S, C = S;

    ll topR1 = 2 + margin + L, topR2 = topR1 + IS - 1;
    ll botR2 = S - 1 - margin - L, botR1 = botR2 - IS + 1;
    ll leftC1 = 2 + margin + L, leftC2 = leftC1 + IS - 1;
    ll rightC2 = S - 1 - margin - L, rightC1 = rightC2 - IS + 1;

    vector<string> grid(R + 1, string(C + 1, '.'));
    for (ll c = 1; c <= C; c++) { grid[1][c] = '#'; grid[R][c] = '#'; }
    for (ll r = 1; r <= R; r++) { grid[r][1] = '#'; grid[r][C] = '#'; }

    vector<ll> ir1(K + 1), ir2(K + 1), ic1(K + 1), ic2(K + 1);
    // quadrant assignment: 1=TL 2=TR 3=BL
    for (int i = 1; i <= K; i++) {
        ll r1, r2, c1, c2;
        if (i == 1) { r1 = topR1; r2 = topR2; c1 = leftC1; c2 = leftC2; }
        else if (i == 2) { r1 = topR1; r2 = topR2; c1 = rightC1; c2 = rightC2; }
        else { r1 = botR1; r2 = botR2; c1 = leftC1; c2 = leftC2; }
        ir1[i] = r1; ir2[i] = r2; ic1[i] = c1; ic2[i] = c2;
        for (ll r = r1; r <= r2; r++)
            for (ll c = c1; c <= c2; c++)
                grid[r][c] = (char)('0' + i);
    }

    // visual cost map: 3 near any structure (border or island, Chebyshev dist<=2), else 1
    vector<string> cost(R + 1, string(C + 1, '1'));
    for (ll r = 1; r <= R; r++)
        for (ll c = 1; c <= C; c++) {
            bool near = false;
            if (r <= 3 || r >= R - 2 || c <= 3 || c >= C - 2) near = true;
            if (!near) {
                for (int i = 1; i <= K && !near; i++) {
                    if (r >= ir1[i] - 2 && r <= ir2[i] + 2 && c >= ic1[i] - 2 && c <= ic2[i] + 2) near = true;
                }
            }
            cost[r][c] = near ? '3' : '1';
        }

    printf("%lld %lld %d %d %d\n", R, C, K, A, BD);
    for (ll r = 1; r <= R; r++) printf("%s\n", grid[r].c_str() + 1);
    for (ll r = 1; r <= R; r++) printf("%s\n", cost[r].c_str() + 1);
    for (int i = 1; i <= K; i++) printf("%d %lld\n", i, Sreq[i]);
    return 0;
}
