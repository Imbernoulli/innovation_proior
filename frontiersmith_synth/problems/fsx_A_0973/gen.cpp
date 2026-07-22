#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Phased-Array Drift-Anchored Calibration Sweep"
// family: drift-anchored-spanning-calibration
//
// Emits: N T ; then N lines x_i y_i d_i  (element 0 = perfect reference, d_0=0).
//
// Test ladder (testId 1..10), difficulty + structure grows:
//   1-2  small/general sanity, mild drift, mild extra time budget.
//   3    TRAP: several "branch seed" nodes hang directly off the anchor; one of
//        them is a high-drift HUB with many cheap leaves clustered around it.
//        A weight-only MST + fixed BFS-level traversal connects the OTHER
//        branch seeds' children before it ever returns to the hub's own
//        children, leaving the hub stale for many steps -> its children pay
//        a huge drift*(elapsed) surcharge. A schedule-aware strategy instead
//        uses the hub for all its children immediately (its clock keeps
//        resetting), avoiding the surcharge entirely.
//   4    general, larger.
//   5    TRAP: parent-choice ambiguity -- several leaves sit geometrically
//        closest to a high-drift node but only slightly farther from a
//        low-drift node; picking purely by static distance (ignores drift)
//        is a trap.
//   6    NEEDLE: most of the field is noisy/expensive: far apart, moderate
//        drift. A small tight low-drift chain (the needle) offers a cheap
//        calibration backbone hidden among the noise.
//   7    general, larger scale.
//   8    TRAP: two independent high-drift hubs whose branches interleave in
//        BFS level order, doubling the stale-hub surcharge opportunity, at
//        larger N.
//   9    general, large scale, moderate extra time budget.
//   10   largest/adversarial: several hubs of very different drift rates,
//        wide drift variance, generous extra time budget (rhythm scheduling
//        + redundancy re-touch both matter), fills the size envelope.
// -----------------------------------------------------------------------------

struct Node { ll x, y, d; };
static vector<Node> nodes;

static void addNode(ll x, ll y, ll d) {
    x = max((ll)0, min((ll)2000, x));
    y = max((ll)0, min((ll)2000, y));
    d = max((ll)1, min((ll)300, d));
    nodes.push_back({x, y, d});
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    nodes.clear();
    nodes.push_back({0, 0, 0});  // element 0: master reference

    ll extra = 0;  // extra time-budget beyond the minimal N-1 connecting steps

    if (testId == 1) {
        // tiny general sanity -- modest drift so MST's weight savings clearly
        // dominate (topology matters, but no adversarial staleness yet).
        int n = 6;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 150), rnd.next(0, 150), rnd.next(1, 3));
        extra = 1;
    } else if (testId == 2) {
        int n = 10;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 400), rnd.next(0, 400), rnd.next(1, 3));
        extra = 2;
    } else if (testId == 3) {
        // TRAP: 4 branch seeds off the anchor; seed[1] is the high-drift hub
        // with many close leaves; the others get 1-2 leaves each. Leaves are
        // interleaved ROUND-ROBIN across branches (not appended contiguously
        // per branch), so a BFS traversal that visits same-depth nodes in
        // ascending index order keeps hopping away from the hub's own leaves
        // between touches -- exactly the "no static tree resembles a
        // time-braided schedule" trap.
        ll ang[4] = {0, 90, 180, 270};
        vector<ll> bx(4), by(4), bd(4);
        int leafCount[4];
        for (int s = 0; s < 4; s++) {
            double rad = ang[s] * 3.14159265358979 / 180.0;
            bx[s] = (ll)llround(700 + 650 * cos(rad));
            by[s] = (ll)llround(700 + 650 * sin(rad));
            bd[s] = (s == 1) ? rnd.next(180, 260) : rnd.next(1, 3);
            leafCount[s] = (s == 1) ? rnd.next(6, 7) : rnd.next(2, 3);
        }
        for (int s = 0; s < 4; s++) addNode(bx[s], by[s], bd[s]);
        int maxLeaves = *max_element(leafCount, leafCount + 4);
        for (int k = 0; k < maxLeaves; k++)
            for (int s = 0; s < 4; s++)
                if (k < leafCount[s])
                    addNode(bx[s] + rnd.next(-15, 15), by[s] + rnd.next(-15, 15), rnd.next(1, 4));
        extra = 3;
    } else if (testId == 4) {
        int n = 20;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 700), rnd.next(0, 700), rnd.next(1, 3));
        extra = 3;
    } else if (testId == 5) {
        // TRAP: parent-choice ambiguity. Anchor plus one high-drift near
        // node H and one low-drift far node L for several groups; leaves are
        // marginally closer to H than to L.
        int groups = 6;
        for (int g = 0; g < groups; g++) {
            ll cx = rnd.next(300, 1700), cy = rnd.next(300, 1700);
            addNode(cx, cy, rnd.next(200, 280));                 // H: near, fast drift
            ll lx = cx + rnd.next(60, 90) * (rnd.next(0, 1) ? 1 : -1);
            ll ly = cy + rnd.next(60, 90) * (rnd.next(0, 1) ? 1 : -1);
            addNode(lx, ly, rnd.next(1, 3));                     // L: a bit farther, slow drift
            int leaves = rnd.next(2, 3);
            for (int k = 0; k < leaves; k++) {
                // placed so w(H,leaf) < w(L,leaf) by a small margin
                addNode(cx + rnd.next(-8, 8), cy + rnd.next(-8, 8), rnd.next(1, 4));
            }
        }
        extra = 4;
    } else if (testId == 6) {
        // NEEDLE: noisy far scatter + a tight cheap low-drift chain.
        int noise = 22;
        for (int i = 0; i < noise; i++)
            addNode(rnd.next(0, 2000), rnd.next(0, 2000), rnd.next(2, 6));
        ll cx = 1000, cy = 1000;
        int chain = 5;
        for (int i = 0; i < chain; i++) {
            addNode(cx + i * 6, cy + i * 6, rnd.next(1, 2));
        }
        extra = 5;
    } else if (testId == 7) {
        int n = 50;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 1200), rnd.next(0, 1200), rnd.next(1, 4));
        extra = 8;
    } else if (testId == 8) {
        // TRAP: two independent high-drift hubs plus three slow-drift branch
        // seeds, at larger scale. Leaves round-robin interleaved across all
        // 5 groups so BFS-by-index keeps hopping away from each hub between
        // its own touches (doubles the stale-hub surcharge opportunity).
        const int G = 5;
        ll gx[G], gy[G], gd[G]; int gleaves[G];
        for (int h = 0; h < 2; h++) {
            gx[h] = 300 + h * 1400; gy[h] = 300 + h * 1400;
            gd[h] = rnd.next(190, 270);
            gleaves[h] = rnd.next(9, 12);
        }
        for (int s = 2; s < G; s++) {
            gx[s] = rnd.next(0, 2000); gy[s] = rnd.next(0, 2000);
            gd[s] = rnd.next(1, 4);
            gleaves[s] = rnd.next(5, 8);
        }
        for (int g = 0; g < G; g++) addNode(gx[g], gy[g], gd[g]);
        int maxLeaves = *max_element(gleaves, gleaves + G);
        for (int k = 0; k < maxLeaves; k++)
            for (int g = 0; g < G; g++)
                if (k < gleaves[g])
                    addNode(gx[g] + rnd.next(-25, 25), gy[g] + rnd.next(-25, 25), rnd.next(1, 6));
        // pad up to ~70 with general noise
        while ((int)nodes.size() < 70)
            addNode(rnd.next(0, 2000), rnd.next(0, 2000), rnd.next(1, 5));
        extra = 12;
    } else if (testId == 9) {
        int n = 110;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 2000), rnd.next(0, 2000), rnd.next(1, 4));
        extra = 15;
    } else if (testId == 10) {
        // largest/adversarial: several hubs with wide drift variance, leaves
        // round-robin interleaved across hubs, plus general scatter, and a
        // generous extra budget so both rhythm scheduling and redundant
        // re-touch matter. Fills the size envelope (N near the 200 cap).
        const int nHubs = 5;
        vector<ll> hx(nHubs), hy(nHubs), hd(nHubs);
        int hleaves[nHubs];
        for (int h = 0; h < nHubs; h++) {
            hx[h] = rnd.next(0, 2000); hy[h] = rnd.next(0, 2000);
            hd[h] = rnd.next(60, 300);          // hubs: genuinely fast drift
            hleaves[h] = rnd.next(15, 22);
        }
        for (int h = 0; h < nHubs; h++) addNode(hx[h], hy[h], hd[h]);
        int maxLeaves = *max_element(hleaves, hleaves + nHubs);
        for (int k = 0; k < maxLeaves; k++)
            for (int h = 0; h < nHubs; h++)
                if (k < hleaves[h])
                    addNode(hx[h] + rnd.next(-25, 25), hy[h] + rnd.next(-25, 25), rnd.next(1, 6));
        while ((int)nodes.size() < 170)
            addNode(rnd.next(0, 2000), rnd.next(0, 2000), rnd.next(1, 5));
        extra = 45;
    } else {
        int n = 12;
        for (int i = 1; i < n; i++) addNode(rnd.next(0, 300), rnd.next(0, 300), rnd.next(1, 10));
        extra = 2;
    }

    int N = (int)nodes.size();
    if (N > 200) { nodes.resize(200); N = 200; }
    ll T = (ll)(N - 1) + extra;
    if (T > (ll)N + 120) T = (ll)N + 120;
    if (T < (ll)N - 1) T = (ll)N - 1;

    printf("%d %lld\n", N, T);
    for (int i = 0; i < N; i++) {
        printf("%lld %lld %lld\n", nodes[i].x, nodes[i].y, nodes[i].d);
    }
    return 0;
}
