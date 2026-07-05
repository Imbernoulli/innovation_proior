#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// Regolith Relay Dispersion generator.
// testId is a difficulty / structure ladder:
//   1 tiny (example scale), then growing to the envelope by 10.
// Structure modes:
//   0 uniform  - hubs fill the basin; a padded empty rim gives dispersion room
//   1 planted  - hubs on a pristine regular lattice (a hidden clean optimum) with
//                the resonance rule aligned so the lattice is highly resonant (needle for Align)
//   2 trap     - hubs crowd one quadrant of a LARGE grid: the naive cover-and-cluster
//                layout clusters, while spreading spare pods into the empty basin wins
//   3 needle   - a dense noisy demand blob plus far-flung hubs; one high-value dispersed
//                structure hides amid the noise
// Invariants ALWAYS enforced => instance is feasible with C (<= m) hub pods:
//   - spacing > R  => every hub is placed by a greedy cover (no hub covers another),
//   - each demand cell lies within L1 radius Roff (<= R) of some hub  => hubs cover all demand,
//   - the first C emitted demand cells are the hubs,  C <= m <= D,  all cells distinct, in range.

int W, H;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int T = atoi(argv[1]);

    int hc, spacing, mode;
    switch (T) {
        case 1:  hc = 2;  spacing = 12; mode = 0; break; // tiny, example-ish scale
        case 2:  hc = 3;  spacing = 20; mode = 0; break;
        case 3:  hc = 5;  spacing = 26; mode = 1; break; // planted resonant lattice
        case 4:  hc = 8;  spacing = 28; mode = 0; break;
        case 5:  hc = 12; spacing = 30; mode = 2; break; // trap
        case 6:  hc = 16; spacing = 30; mode = 0; break;
        case 7:  hc = 22; spacing = 30; mode = 3; break; // needle
        case 8:  hc = 30; spacing = 26; mode = 2; break; // trap, larger
        case 9:  hc = 40; spacing = 26; mode = 1; break; // planted, large
        default: hc = 50; spacing = 24; mode = 0; break; // t=10 large envelope
    }

    int Roff = max(3, spacing / 2);          // demand offset annulus outer radius
    int minoff = max(2, Roff / 2);           // inner radius: keeps trivial's Dmin off 1
    int R = min(400, (spacing * 3) / 4);      // coverage radius: Roff <= R < spacing
    if (R <= Roff) R = Roff + 1;

    int hubSpan = hc;
    int gridSide = (int)llround(hc * 1.6);   // padded empty rim for dispersion
    if (mode == 2) gridSide = hc * 2;        // trap: big empty basin
    if (mode == 3) gridSide = hc + 6;
    if (gridSide < hubSpan + 1) gridSide = hubSpan + 1;

    W = gridSide * spacing;
    H = gridSide * spacing;
    if (W < 4) W = 4;
    if (H < 4) H = 4;
    if (W > 2600) W = 2600;
    if (H > 2600) H = 2600;

    // resonance rule
    int a = 2, b = 1, K = 3, r0;
    if (mode == 1) { a = 1; b = 1; K = 3; r0 = 0; } // planted: lattice tends resonant
    else           { a = 2; b = 1; K = 3; r0 = rnd.next(0, K - 1); }

    // ---- hubs ----
    set<pair<int,int>> occ;
    vector<pair<int,int>> hubs;
    auto add = [&](int x, int y, vector<pair<int,int>>& v) -> bool {
        x = max(0, min(W - 1, x));
        y = max(0, min(H - 1, y));
        if (occ.count({x, y})) return false;
        occ.insert({x, y});
        v.push_back({x, y});
        return true;
    };
    for (int gi = 0; gi < hubSpan; gi++)
        for (int gj = 0; gj < hubSpan; gj++) {
            int cx = (int)llround((gi + 0.5) * spacing);
            int cy = (int)llround((gj + 0.5) * spacing);
            if (mode == 1) add(cx, cy, hubs);               // pristine lattice
            else {
                int jr = spacing / 5;
                add(cx + rnd.next(-jr, jr), cy + rnd.next(-jr, jr), hubs);
            }
        }
    int C = (int)hubs.size();

    int m = C + max(1, C);                   // ~2C pods: half cover, half spare
    int extraTarget = 3 * C + 5;
    int noiseHub = (mode == 3 && C > 0) ? rnd.next(0, C - 1) : -1;

    vector<pair<int,int>> demand = hubs;     // hubs first
    int attempts = 0, maxAttempts = extraTarget * 40 + 200;
    while ((int)demand.size() < C + extraTarget && attempts < maxAttempts) {
        attempts++;
        int h = (noiseHub >= 0 && rnd.next(0, 3) != 0) ? noiseHub : rnd.next(0, C - 1);
        // sample an L1 offset in the annulus [minoff, Roff]
        int rr = rnd.next(minoff, Roff);
        int ox = rnd.next(-rr, rr);
        int rem = rr - abs(ox);
        int oy = rnd.next(-rem, rem);
        if (abs(ox) + abs(oy) < minoff) { oy += (oy >= 0 ? 1 : -1); } // nudge off the core
        add(hubs[h].first + ox, hubs[h].second + oy, demand);
    }
    int D = (int)demand.size();

    if (m > D) m = D;
    if (m < C) m = C;
    if (m < 2) m = min(2, D);

    printf("%d %d %d %d %d %d %d %d %d\n", W, H, m, R, D, a, b, K, r0);
    for (int i = 0; i < D; i++)
        printf("%d %d\n", demand[i].first, demand[i].second);
    return 0;
}
