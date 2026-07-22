#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Dispersed Relay Coverage Under a Roaming Jammer"   (generator)
// family: jammer-grid-coverage     objective: MAXIMIZE
//
// Mechanisms composed into one objective:
//   (1) weighted-disc-coverage   -- point i worth w_i, needs t_i (1..3) surviving
//                                    towers within r for full (saturating) credit.
//   (2) deletion-disc-adversary  -- the checker takes the MIN over M published
//                                    jammer positions; every tower within R of the
//                                    active jammer is destroyed before scoring.
//   (3) anti-clustering-redundancy -- because only the worst jammer position counts,
//                                    piling redundant towers inside the reach of ONE
//                                    jammer candidate buys nothing; value only survives
//                                    if spread so no single candidate threatens a
//                                    critical mass of it.
//
// testId is a difficulty/structure ladder:
//   1  tiny uniform sanity
//   2  CLUSTER TRAP: one tight heavy cluster (diameter < R) + light decoys; a jammer
//      candidate sits exactly at the cluster centroid.
//   3  RESILIENCE TRAP: several t=3 points close together (diameter < R) so internal
//      redundancy is worthless; must borrow survivors from elsewhere.
//   4  medium uniform (no crafted trap -- sanity that strong doesn't overfit traps).
//   5  MULTI-CLUSTER TRAP: three clusters of very different total weight, jammer
//      candidates parked at each centroid; budget forces a real allocation choice.
//   6  NEEDLE: one very high-weight point buried in cheap noise, dense local jammer
//      ring around it.
//   7  larger uniform.
//   8  FULL-COVERAGE TRAP: several clusters AND a coarse jammer grid tiling the
//      whole map, so there is no safe empty region to hide spare towers in either.
//   9  RING TRAP: demand arranged on an arc/ring with jammers also seeded along the
//      ring, so piling on the richest arc (ignoring the rest) is punished.
//   10 max-envelope mixed stress test (clusters + ring + noise, largest N/K/M/L).
// -----------------------------------------------------------------------------

int L;
vector<array<int,4>> dem;      // x y w t
vector<pair<int,int>> jam;     // gx gy

static int clampL(int v){ return max(0, min(L, v)); }

void addDemand(int x, int y, int w, int t){
    dem.push_back({clampL(x), clampL(y), w, t});
}
void addJammer(int x, int y){
    jam.push_back({clampL(x), clampL(y)});
}

// spread `count` points uniformly at random inside a disc of given radius around (cx,cy);
// returns the generated coordinates so callers can plant an escape point that targets one.
vector<pair<int,int>> genClusterPoints(int cx, int cy, int radius, int count, int wLo, int wHi, int t){
    vector<pair<int,int>> out;
    for (int i = 0; i < count; i++){
        double ang = rnd.next(0.0, 2.0 * acos(-1.0));
        double rad = radius * sqrt(rnd.next(0.0, 1.0));
        int x = cx + (int)llround(rad * cos(ang));
        int y = cy + (int)llround(rad * sin(ang));
        addDemand(x, y, rnd.next(wLo, wHi), t);
        out.push_back({x, y});
    }
    return out;
}

// Plant a demand point that is a genuine "escape" position: it lies within tower
// coverage radius `rr` of a rich target point (tx,ty) inside a jammed cluster, but
// beyond the jamming radius `Rr` of the jammer sitting at (jx,jy) -- reachable and
// safe at once, since rr > Rr for these tests. A naive one-pass greedy that already
// "spent" its per-point marginal budget placing towers directly on the cluster never
// discovers that THIS location reaches the same rich point while surviving the very
// jammer that guards it.
void addEscapePoint(int jx, int jy, int tx, int ty, int Rr, int rr, int w, int t){
    double dx = tx - jx, dy = ty - jy;
    double dist = sqrt(dx * dx + dy * dy);
    double ux, uy;
    if (dist < 1e-6){ ux = 1.0; uy = 0.0; dist = 0.0; } else { ux = dx / dist; uy = dy / dist; }
    double extra = Rr - dist + 3.0;             // margin past the jamming boundary
    if (extra < 1.0) extra = 1.0;
    if (extra > rr - 2.0) extra = rr - 2.0;      // stay safely within coverage radius
    if (extra < 1.0) extra = 1.0;
    int x = tx + (int)llround(extra * ux);
    int y = ty + (int)llround(extra * uy);
    addDemand(x, y, w, t);
}

void genUniformPoints(int count, int wLo, int wHi, int tLo, int tHi){
    for (int i = 0; i < count; i++)
        addDemand(rnd.next(0, L), rnd.next(0, L), rnd.next(wLo, wHi), rnd.next(tLo, tHi));
}

void genGridJammers(int per_axis){
    for (int i = 0; i < per_axis; i++)
        for (int j = 0; j < per_axis; j++){
            int x = (int)((ll)(2 * i + 1) * L / (2 * per_axis));
            int y = (int)((ll)(2 * j + 1) * L / (2 * per_axis));
            addJammer(x, y);
        }
}

void genRandomJammers(int count){
    for (int i = 0; i < count; i++) addJammer(rnd.next(0, L), rnd.next(0, L));
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int K, r, R;

    if (testId == 1){
        // tiny uniform sanity
        L = 100; K = 3; r = 20; R = 15;
        genUniformPoints(6, 5, 100, 1, 2);
        genRandomJammers(4);
    } else if (testId == 2){
        // CLUSTER TRAP #1: one tight heavy cluster + scattered light decoys
        L = 200; K = 6; r = 22; R = 18;
        int cx = 60, cy = 60, crad = 10;           // diameter ~20 < R=18*2, well inside one disc
        auto c1 = genClusterPoints(cx, cy, crad, 8, 150, 400, 2);
        genUniformPoints(10, 3, 20, 1, 1);          // light decoys spread over the map
        addJammer(cx, cy);                          // worst case parked exactly on the cluster
        genGridJammers(3);
        addEscapePoint(cx, cy, c1[0].first, c1[0].second, R, r, 18, 1);   // planted escape
    } else if (testId == 3){
        // RESILIENCE TRAP: t=3 points crammed together, internal redundancy is worthless
        L = 200; K = 8; r = 20; R = 16;
        int cx = 100, cy = 100, crad = 8;
        genClusterPoints(cx, cy, crad, 6, 200, 500, 3);
        genUniformPoints(12, 5, 30, 1, 1);
        addJammer(cx, cy);
        genGridJammers(3);
        genRandomJammers(3);
    } else if (testId == 4){
        // medium uniform, no crafted trap
        L = 300; K = 10; r = 28; R = 22;
        genUniformPoints(40, 5, 200, 1, 2);
        genRandomJammers(20);
    } else if (testId == 5){
        // MULTI-CLUSTER TRAP: three clusters, very different weight, budget forces a choice
        L = 300; K = 10; r = 26; R = 20;
        auto c1 = genClusterPoints(60, 60, 9, 10, 50, 85, 3);    // rich cluster
        auto c2 = genClusterPoints(240, 60, 9, 8, 14, 26, 2);    // medium cluster
        auto c3 = genClusterPoints(150, 240, 9, 6, 4, 11, 1);    // poor cluster
        genUniformPoints(21, 3, 15, 1, 1);
        addJammer(60, 60); addJammer(240, 60); addJammer(150, 240);
        genGridJammers(4);
        addEscapePoint(60, 60, c1[0].first, c1[0].second, R, r, 20, 1);
        addEscapePoint(240, 60, c2[0].first, c2[0].second, R, r, 12, 1);
    } else if (testId == 6){
        // NEEDLE: one very high-weight point buried in cheap noise + dense local jammer ring
        L = 300; K = 10; r = 24; R = 20;
        addDemand(150, 150, 900, 3);                    // the needle
        genUniformPoints(49, 2, 12, 1, 1);               // cheap noise everywhere
        for (int k = 0; k < 8; k++){                     // dense local jammer ring around the needle
            double ang = 2.0 * acos(-1.0) * k / 8.0;
            addJammer(150 + (int)llround(14 * cos(ang)), 150 + (int)llround(14 * sin(ang)));
        }
        addJammer(150, 150);
        genGridJammers(4);
    } else if (testId == 7){
        // larger uniform
        L = 400; K = 16; r = 30; R = 24;
        genUniformPoints(70, 5, 250, 1, 2);
        genRandomJammers(45);
    } else if (testId == 8){
        // FULL-COVERAGE TRAP: several clusters + a coarse jammer grid tiling the whole map
        L = 400; K = 18; r = 28; R = 22;
        auto c1 = genClusterPoints(80, 80, 10, 12, 35, 60, 3);
        auto c2 = genClusterPoints(320, 80, 10, 10, 22, 40, 2);
        auto c3 = genClusterPoints(80, 320, 10, 10, 22, 40, 2);
        auto c4 = genClusterPoints(320, 320, 10, 10, 14, 28, 2);
        genUniformPoints(43, 3, 20, 1, 1);
        addJammer(80, 80); addJammer(320, 80); addJammer(80, 320); addJammer(320, 320);
        genGridJammers(7);   // 49 more, tiles the whole map -- no safe empty region either
        genRandomJammers(3);
        addEscapePoint(80, 80, c1[0].first, c1[0].second, R, r, 16, 1);
        addEscapePoint(320, 80, c2[0].first, c2[0].second, R, r, 14, 1);
        addEscapePoint(80, 320, c3[0].first, c3[0].second, R, r, 14, 1);
        addEscapePoint(320, 320, c4[0].first, c4[0].second, R, r, 12, 1);
    } else if (testId == 9){
        // RING TRAP: demand on an arc/ring, jammers seeded along the ring too
        L = 450; K = 20; r = 26; R = 20;
        int cx = 225, cy = 225, ring = 160;
        int nring = 60;
        vector<pair<int,int>> ringPts(nring);
        for (int k = 0; k < nring; k++){
            double ang = 2.0 * acos(-1.0) * k / nring;
            int x = cx + (int)llround(ring * cos(ang));
            int y = cy + (int)llround(ring * sin(ang));
            // weight skewed: one rich arc (k in [0,15)) tempts the greedy to camp there
            int w = (k < 15) ? rnd.next(12, 26) : rnd.next(4, 16);
            int t = (k < 15) ? 3 : rnd.next(1, 2);
            addDemand(x, y, w, t);
            ringPts[k] = {x, y};
        }
        genUniformPoints(40, 3, 20, 1, 1);
        vector<pair<int,int>> ringJam;
        for (int k = 0; k < nring; k += 5){
            double ang = 2.0 * acos(-1.0) * k / nring;
            int jxp = cx + (int)llround(ring * cos(ang)), jyp = cy + (int)llround(ring * sin(ang));
            addJammer(jxp, jyp);
            ringJam.push_back({jxp, jyp});
        }
        genGridJammers(4);
        // the rich arc points at k=0,5,10 sit exactly ON a ring jammer; plant an escape
        // point reaching each while surviving that jammer.
        addEscapePoint(ringJam[0].first, ringJam[0].second, ringPts[0].first, ringPts[0].second, R, r, 20, 1);
        addEscapePoint(ringJam[1].first, ringJam[1].second, ringPts[5].first, ringPts[5].second, R, r, 20, 1);
        addEscapePoint(ringJam[2].first, ringJam[2].second, ringPts[10].first, ringPts[10].second, R, r, 20, 1);
    } else {
        // testId == 10: max-envelope mixed stress test
        L = 500; K = 30; r = 30; R = 24;
        genClusterPoints(100, 100, 12, 14, 250, 500, 3);
        genClusterPoints(400, 100, 12, 12, 150, 350, 2);
        genClusterPoints(100, 400, 12, 12, 150, 350, 2);
        genClusterPoints(400, 400, 12, 12, 100, 250, 2);
        int cx = 250, cy = 250, ring = 180, nring = 30;
        for (int k = 0; k < nring; k++){
            double ang = 2.0 * acos(-1.0) * k / nring;
            int x = cx + (int)llround(ring * cos(ang));
            int y = cy + (int)llround(ring * sin(ang));
            addDemand(x, y, rnd.next(40, 200), rnd.next(1, 3));
        }
        genUniformPoints(30, 3, 20, 1, 1);
        addJammer(100, 100); addJammer(400, 100); addJammer(100, 400); addJammer(400, 400);
        addJammer(cx, cy);
        genGridJammers(9);   // 81 more
        genRandomJammers(15);
    }

    int N = (int)dem.size();
    int M = (int)jam.size();

    printf("%d %d %d %d %d %d\n", N, K, M, L, r, R);
    for (auto& p : dem) printf("%d %d %d %d\n", p[0], p[1], p[2], p[3]);
    for (auto& g : jam) printf("%d %d\n", g.first, g.second);
    return 0;
}
