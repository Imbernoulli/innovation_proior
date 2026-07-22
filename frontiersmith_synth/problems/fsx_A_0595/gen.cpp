#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Order Slab"  (generator)  family: guillotine-cut-tree-planner
//
// A sheet W x H is carved into a GUILLOTINE cut tree (no kerf: a cut splits one
// rectangle edge-to-edge into two whose sizes sum exactly). A leaf may be sold as
// type t ONLY when its size equals (w_t,h_t) EXACTLY; type t sells at most c_t times.
// Maximize sum of realized values. Fragments never rejoin: once a band is cut to
// height h, every piece of it is frozen at height h forever.
//
// The value of a plan is set by the LEAF-SIZE DISTRIBUTION of the whole tree, not by
// the piece a cut immediately yields. The instance is built to make PER-PIECE VALUE
// anti-correlate with VALUE-DENSITY:
//   * HERO types: wide (only one fits across the sheet, w > W/2), TOP per-piece value,
//     but LOW density -> a hero piece wastes most of its band. Its per-band total
//     (value * 1) stays below a full dense band, so the checker baseline B (best
//     single shelf) is a DENSE band, not a hero.
//   * DENSE types: narrow, high density, each sellable for about one full band. They
//     are what a good plan fills bands with.
// A value-first greedy opens a band for the highest PER-PIECE value (the hero) and
// wastes the band; a density-aware plan fills bands with dense pieces for far more.
// The trap bites on every hero band, not just once.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    rnd.setSeed(testId * 1000003LL + 777LL);

    int W = 60 + 10 * testId;   if (W > 160) W = 160;
    int H = 40 + 8 * testId;    if (H > 130) H = 130;

    vector<int> palette = {8, 10, 12, 14};
    int nH = 2 + testId / 3;    if (nH > (int)palette.size()) nH = palette.size();

    struct T { int w, h, v, c; };
    vector<T> types;

    int maxDensePiece = 1, maxDenseBand = 1;
    // dense narrow types: high density, cap = one band each (bounds strong's total)
    for (int hi = 0; hi < nH; hi++){
        int h = palette[hi];
        int nDense = 1 + rnd.next(0, 1);
        for (int k = 0; k < nDense; k++){
            int w = rnd.next(8, max(9, W / 6));      // narrow -> many fit per band
            int dens = rnd.next(10, 16);             // high value-density
            int v = max(1, dens * w * h / 10);
            int fit = W / w;
            int cap = fit;                           // exactly one full band of supply
            types.push_back({w, h, v, cap});
            maxDensePiece = max(maxDensePiece, v);
            maxDenseBand  = max(maxDenseBand,  v * fit);
        }
    }

    // hero types: wide, top per-piece value, low density -> greedy bait
    int nHero = 1 + (testId >= 5 ? rnd.next(0, 1) : 0);
    for (int s = 0; s < nHero; s++){
        int h = palette[rnd.next(0, nH - 1)];
        int w = rnd.next((W * 11) / 20, (W * 7) / 10);   // > W/2 -> only one fits
        if (w >= W) w = W - 1;
        // per-piece value tops every dense piece, but a hero band (v*1) < a dense band
        int lo = (maxDensePiece * 13) / 10 + 1;
        int hi = max(lo + 1, (maxDenseBand * 6) / 10);
        int v = rnd.next(lo, hi);
        int cap = rnd.next(4, 9);                        // wastes MANY bands under greedy
        types.push_back({w, h, v, cap});
    }

    // a little low-value noise (needle amid noise)
    int nNoise = 1 + testId / 2;
    for (int i = 0; i < nNoise; i++){
        int h = palette[rnd.next(0, nH - 1)];
        int w = rnd.next(6, max(8, W / 5));
        int v = rnd.next(1, 3) * w;                       // low density, low value
        int cap = rnd.next(1, 4);
        types.push_back({w, h, v, cap});
    }

    shuffle(types.begin(), types.end());
    int D = (int)types.size();
    int K = 3000;

    printf("%d %d %d %d\n", W, H, K, D);
    for (auto &tp : types)
        printf("%d %d %d %d\n", tp.w, tp.h, tp.v, tp.c);
    return 0;
}
