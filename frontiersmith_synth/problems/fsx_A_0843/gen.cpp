#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

// -----------------------------------------------------------------------------
// "Steamworks Deck: Cold-Rail Block Stowage"  (generator)
// family: hot-block-floorplan-thermal
//
// A W x H deck whose OUTER RIM is a fixed-temperature (0) heat sink. N polyomino
// blocks, each dissipating w_i watts split over its cells, must be packed
// (rotations allowed, no overlap) so that after ITERS steps of integer Jacobi
// diffusion (rim clamped to 0) the PEAK temperature is minimized.
//
// Physics: distance-to-rim is the first-order term (image-charge / Green's
// function intuition for a grounded boundary) -- a source flush on the rim
// contributes literally nothing (rim cells never update away from 0), while a
// source near the deck center has nowhere close to dump heat and drives the
// peak up a lot for the same wattage. Regimes below deliberately create TRAP
// cases (one dominant "needle" block among many tiny ones, or many similarly
// hot blocks competing for a rim whose capacity is smaller than total block
// area) where a "spread blocks apart" heuristic (which defaults the FIRST,
// heaviest block to the geometric center, since nothing is placed yet to be
// far from) lands far from a rim-hugging, perimeter-spread placement.
// -----------------------------------------------------------------------------

struct Shape { vector<pii> cells; }; // already normalized: min dx=0, min dy=0

static vector<Shape> SHAPE_LIB = {
    {{{0,0}}},                                   // 0 monomino            k=1
    {{{0,0},{1,0}}},                              // 1 domino              k=2
    {{{0,0},{1,0},{2,0}}},                         // 2 tromino-I           k=3
    {{{0,0},{1,0},{0,1}}},                         // 3 tromino-L           k=3
    {{{0,0},{1,0},{0,1},{1,1}}},                   // 4 tetromino-square    k=4
    {{{0,0},{1,0},{2,0},{1,1}}},                   // 5 tetromino-T         k=4
    {{{1,0},{2,0},{0,1},{1,1}}},                   // 6 tetromino-S         k=4
    {{{1,0},{0,1},{1,1},{2,1},{1,2}}},             // 7 pentomino-plus      k=5
};

struct Block { ll w; Shape shp; };

// bounding box at rotation 0
static void bbox(const Shape& s, int& bw, int& bh){
    int mx = 0, my = 0;
    for (auto& c : s.cells){ mx = max(mx, c.first); my = max(my, c.second); }
    bw = mx + 1; bh = my + 1;
}

// simulate left-to-right shelf packing at rotation 0; returns the total
// height used (max cursor_y + row_h reached). Also reports whether every
// placement fits within a given H (feasibility check while sizing).
static int shelfHeight(int W, const vector<Block>& blocks){
    int cursor_x = 0, cursor_y = 0, row_h = 0, used_h = 0;
    for (auto& b : blocks){
        int bw, bh; bbox(b.shp, bw, bh);
        if (cursor_x + bw > W){ cursor_x = 0; cursor_y += row_h; row_h = 0; }
        cursor_x += bw;
        row_h = max(row_h, bh);
        used_h = max(used_h, cursor_y + row_h);
    }
    return used_h;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- pick block count, candidate width, and a wattage regime per testId ----
    int N; int Wc; string regime;
    switch (testId){
        case 1:  N = 3;  Wc = 9;  regime = "normal";     break;
        case 2:  N = 4;  Wc = 10; regime = "normal";     break;
        case 3:  N = 5;  Wc = 12; regime = "planted";    break;
        case 4:  N = 6;  Wc = 13; regime = "planted";    break;
        case 5:  N = 7;  Wc = 14; regime = "normal";     break;
        case 6:  N = 8;  Wc = 16; regime = "needle";     break;
        case 7:  N = 9;  Wc = 17; regime = "needle";     break;
        case 8:  N = 11; Wc = 19; regime = "perimlim";   break;
        case 9:  N = 13; Wc = 23; regime = "needle";     break;
        default: N = 16; Wc = 27; regime = "combined";   break; // testId 10
    }

    vector<Block> blocks;
    for (int i = 0; i < N; i++){
        int shapeId = rnd.next(0, (int)SHAPE_LIB.size() - 1);
        ll w;
        if (regime == "normal"){
            w = 5 + rnd.next(0, 55);                       // 5..60
        } else if (regime == "planted"){
            w = 10 + rnd.next(0, 50);                      // 10..60
        } else if (regime == "needle"){
            if (i == 0) w = 500 + rnd.next(0, 300);        // one dominant block: 500..800
            else        w = 3 + rnd.next(0, 12);            // rest tiny: 3..15
        } else if (regime == "perimlim"){
            w = 70 + rnd.next(0, 30);                       // 70..100, all similarly hot
        } else { // combined: two needles + several perimeter-competitive blocks
            if (i == 0 || i == 1) w = 600 + rnd.next(0, 300); // 600..900
            else                   w = 40 + rnd.next(0, 45);  // 40..85
        }
        blocks.push_back({w, SHAPE_LIB[shapeId]});
    }

    // shuffle listing order so wattage is NOT correlated with input position
    // (keeps the shelf-packing baseline from accidentally favoring the hottest
    // block just because it happens to land in the first, rim-adjacent shelf slot)
    for (int i = (int)blocks.size() - 1; i > 0; i--)
        swap(blocks[i], blocks[rnd.next(0, i)]);

    // ---- size the deck: W = Wc, H = shelf-pack height scaled well past what
    // shelf packing needs. Shelf packing fills rows starting at y=0 (the rim),
    // so with a TIGHT height most of its rows would end up accidentally near
    // the top or bottom rim -- a poor "trivial" reference by accident of
    // geometry, not because it ignores the physics. Tripling the height means
    // most shelf rows land deep in the interior (a genuinely bad placement),
    // while still leaving total block area comfortably above rim capacity so a
    // full rim-only solve is not achievable either -- real interior placement,
    // and real headroom above the reference solvers, is unavoidable. ----
    int W = Wc;
    int h0 = shelfHeight(W, blocks);
    int H = h0 * 3 + 6;
    if (H > 34) H = 34;
    if (H < 9) H = 9;
    if (W < 9) W = 9;
    if (W > 34) W = 34;

    // ITERS: enough Jacobi sweeps to let boundary distance clearly dominate
    // (diffusion mixing time on an L x L grid is O(L^2); this comfortably
    // covers it while staying fast: worst case ~34x34 grid -> ITERS ~ 60*68 = 4080).
    int ITERS = 60 * (W + H);

    printf("%d %d %d %d\n", W, H, N, ITERS);
    for (auto& b : blocks){
        printf("%lld %d\n", b.w, (int)b.shp.cells.size());
        for (size_t j = 0; j < b.shp.cells.size(); j++){
            printf("%d %d%c", b.shp.cells[j].first, b.shp.cells[j].second,
                   j + 1 == b.shp.cells.size() ? '\n' : ' ');
        }
    }
    return 0;
}
