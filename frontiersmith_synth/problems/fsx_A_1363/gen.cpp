#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Bidders who want bundles or nothing"  (generator)
// family: combinatorial-auction-wdp
//
// Composes three mechanisms into one winner-determination instance, built as
// disjoint BLOCKS of bids (each block uses its own private range of item ids,
// so blocks never compete with each other and the item-sharing graph of the
// whole instance decomposes exactly into these blocks):
//
//   1. bundle-complementarity : HUB blocks -- a bundle bid over {hub+petals}
//      competes against a lone "spoiler" bid on just the hub plus small filler
//      bids on the petals. The spoiler alone has a strictly higher per-item
//      price than the bundle can ever reach (by construction), so any
//      density-sorted scan claims the hub before it ever reaches the bundle
//      and permanently kills it, even though bundle > spoiler+fillers.
//   2. winner-determination-np-hard : the whole instance is a 0/1 selection of
//      bids under per-item capacity constraints (weighted set packing);
//      POOL blocks add plain single-item multi-bidder competition (whoever
//      pays most for a scarce item), and a general background of overlapping
//      bundle bids over shared-capacity items adds volume and realistic
//      interaction between unrelated bidders.
//   3. lp-relaxation-gap : CYCLE blocks -- odd rings of uniform-price 2-item
//      bids. The LP relaxation of "accept x_i in [0,1] per bid" can set every
//      ring edge to 1/2 (using each item's capacity exactly), scoring 3/2x the
//      true integral optimum for a triangle -- the classic odd-cycle
//      set-packing integrality-gap witness -- concrete evidence that a
//      fractional relaxation is loose exactly on tightly-overlapping rings
//      and nowhere else in this instance.
//
// A solver that decomposes the instance by shared-item connectivity finds
// every hub/path/pool/ring as its own tiny separate component and can afford
// to branch (brute force) each one exactly, while the loose background is
// cheap to handle with a plain heuristic. A flat density-sorted greedy cannot
// see this decomposition and is systematically trapped in every hub and path
// block, in every test case.
//
// Each block is emitted with a fixed INTERNAL bid order (documented at each
// site) so its outcome under a pure "take the next bid if it still fits"
// scan is deterministic, never a matter of luck; the checker's own baseline
// scans bids in exactly the input order. Only the order BETWEEN blocks is
// shuffled (never within a block), since cross-block order cannot affect any
// block's own outcome -- blocks share no items.
// -----------------------------------------------------------------------------

struct Bid { int k; ll p; vector<int> items; };
typedef vector<Bid> Block;

int nextItem = 1;
vector<ll> capv;
int newItem(ll c) { capv.push_back(c); return nextItem++; }

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    capv.push_back(0); // dummy index 0
    vector<Block> blocks;

    // ---------------- background: general overlapping bundle bids ----------------
    // Kept deliberately small relative to the gadgets below: with ample
    // per-item supply this block is captured about equally well by any
    // strategy, so it must not dilute the trap signal the gadgets carry.
    {
        int bgItems = 5 + (int)llround(f * 15.0);      // 5..20
        int bgBidCount = 8 + (int)llround(f * 25.0);   // 8..33
        vector<int> bgId(bgItems);
        for (int i = 0; i < bgItems; i++) bgId[i] = newItem(2 + rnd.next(0, 4)); // cap 2..6
        Block blk;
        for (int i = 0; i < bgBidCount; i++) {
            int k = min(1 + rnd.next(0, 2), bgItems); // 1..3
            vector<int> chosen; set<int> used;
            while ((int)chosen.size() < k) {
                int idx = rnd.next(0, bgItems - 1);
                if (used.count(idx)) continue;
                used.insert(idx);
                chosen.push_back(bgId[idx]);
            }
            ll p = 8 + rnd.next(0, 37); // 8..45
            blk.push_back({k, p, chosen});
        }
        blocks.push_back(blk);
    }

    // ---------------- HUB blocks: bundle-complementarity trap ----------------
    // Internal order: spoiler, then every filler, then the bundle LAST.
    // Guarantee: margin is capped below S*petals - sumFillers, which is
    // exactly the algebraic condition for the bundle's per-item price
    // (price / bundle size) to stay BELOW the spoiler's price -- so a
    // density-sorted greedy always ranks the spoiler ahead of the bundle and
    // claims the hub first, killing the bundle, on every single hub.
    int nHub = 6 + (int)llround(f * 17.0); // 6..23
    for (int h = 0; h < nHub; h++) {
        int petals = 3 + rnd.next(0, 3); // 3..6
        int hub = newItem(1);
        vector<int> petalIds(petals);
        vector<ll> fillerPrice(petals);
        ll sumF = 0;
        for (int j = 0; j < petals; j++) {
            petalIds[j] = newItem(1);
            fillerPrice[j] = 18 + rnd.next(0, 14); // 18..32
            sumF += fillerPrice[j];
        }
        ll avgF = sumF / petals;
        ll S = 4 * avgF + 10 + rnd.next(0, 10);             // S far above the average filler
        ll cap_margin = S * (ll)petals - sumF;              // > 0 since S > sumF/petals
        ll margin = max(1LL, (ll)llround(cap_margin * (0.85 + 0.1 * rnd.next(0, 100) / 100.0)));
        ll Pbundle = S + sumF + margin;                      // bundle strictly beats spoiler+fillers

        Block blk;
        blk.push_back({1, S, {hub}});                        // spoiler first
        for (int j = 0; j < petals; j++)
            blk.push_back({1, fillerPrice[j], {petalIds[j]}}); // fillers
        vector<int> bundleItems = petalIds; bundleItems.push_back(hub);
        blk.push_back({(int)bundleItems.size(), Pbundle, bundleItems}); // bundle last
        blocks.push_back(blk);
    }

    // ---------------- PATH blocks: weighted-matching trap ----------------
    // Internal order: the middle (spoiler) edge first, then the two ends.
    // w2 is just barely above W (still the single highest price in the
    // block) but far below w1+w3=2W -> a greedy/arrival scan that takes the
    // first fitting bid it sees claims the middle edge and blocks both ends,
    // losing close to half of the true optimum.
    int nPath = 6 + (int)llround(f * 17.0); // 6..23
    for (int r = 0; r < nPath; r++) {
        int a = newItem(1), b = newItem(1), c = newItem(1), d = newItem(1);
        ll W = 15 + rnd.next(0, 20); // 15..35
        ll w1 = W, w3 = W;
        ll w2 = W + 1 + rnd.next(0, (int)max(1LL, W / 8)); // W+1 .. ~1.15W
        Block blk;
        blk.push_back({2, w2, {b, c}}); // spoiler first
        blk.push_back({2, w1, {a, b}});
        blk.push_back({2, w3, {c, d}});
        blocks.push_back(blk);
    }

    // ---------------- POOL blocks: plain scarce-item competition ----------------
    // A single item, several independent bidders wanting it alone, prices
    // strictly increasing and emitted in ASCENDING order -- so a scan that
    // "takes the first bid that still fits" always claims the CHEAPEST
    // bidder while a density/price-sorted scan always claims the priciest
    // one. This gives the sorted heuristic a large, deterministic,
    // structural edge with no bundle logic involved at all.
    int nPool = 5 + (int)llround(f * 12.0); // 5..17
    for (int r = 0; r < nPool; r++) {
        int m = 4 + rnd.next(0, 4); // 4..8 competing bidders
        int item = newItem(1);
        ll p0 = 10 + rnd.next(0, 15);           // 10..25
        ll step = 18 + rnd.next(0, 22);         // 18..40
        Block blk;
        for (int j = 0; j < m; j++) {
            ll p = p0 + (ll)j * step + rnd.next(0, 5);
            blk.push_back({1, p, {item}});      // strictly increasing prices, ascending order
        }
        blocks.push_back(blk);
    }

    // ---------------- CYCLE blocks: LP-relaxation-gap structure ----------------
    int nCyc = 1 + (int)llround(f * 3.0); // 1..4
    int kOpts[3] = {3, 5, 7};
    for (int r = 0; r < nCyc; r++) {
        int k = kOpts[rnd.next(0, 2)];
        vector<int> ring(k);
        for (int j = 0; j < k; j++) ring[j] = newItem(1);
        ll w = 12 + rnd.next(0, 20); // 12..32, uniform weight -> genuine fractional LP optimum
        Block blk;
        for (int j = 0; j < k; j++)
            blk.push_back({2, w, {ring[j], ring[(j + 1) % k]}});
        blocks.push_back(blk);
    }

    // ---------------- shuffle BLOCK order only (never within a block) ----------------
    int nb = (int)blocks.size();
    for (int i = nb - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(blocks[i], blocks[j]);
    }
    vector<Bid> bids;
    for (auto &blk : blocks) for (auto &bd : blk) bids.push_back(bd);

    int N = (int)bids.size();
    int M = nextItem - 1;
    printf("%d %d\n", M, N);
    for (int j = 1; j <= M; j++) printf("%lld%c", capv[j], j == M ? '\n' : ' ');
    for (auto &bd : bids) {
        printf("%d %lld", bd.k, bd.p);
        for (int it : bd.items) printf(" %d", it);
        printf("\n");
    }
    return 0;
}
