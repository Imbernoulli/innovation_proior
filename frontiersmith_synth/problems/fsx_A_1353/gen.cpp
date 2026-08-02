// generator for "Sandpile Relief Under a Firing Budget" (fsx_A_1353)
//
// Builds a forest of small independent "clusters" hanging off a shared sink:
//   - pendant cluster: one pile v, one edge (v,sink,w)
//   - chain2 cluster:  two piles v1,v2; edges (v1,sink,w0) and (v1,v2,w1)
// Two deliberate "flavors" of pendant pile create the trap:
//   - flavor H ("loud, expensive"): tiny degree, huge chip count -> huge RAW
//     height (always looks biggest to a naive solver) but a huge odometer
//     cost (firings needed to fully stabilize it), so it is essentially
//     un-fundable within the test's budget.
//   - flavor L ("quiet, cheap"): large degree, moderate chip count -> small
//     raw height (a naive solver ranks it low) but a tiny odometer cost and
//     a large value/cost density (criticality-weighted excess per firing).
// The budget K is sized from the CHEAP (L + chain) cost only, plus a little
// slack -- comfortably funding every cheap pile in a test but nowhere near
// what even a single flavor-H pile needs. So "always fire the biggest
// current pile" (raw height) burns the whole budget on H piles for almost
// no benefit, while ranking by value/cost density funds nearly all of the
// test's total risk away.
#include "testlib.h"
#include <vector>
#include <array>
#include <cstdio>
using namespace std;
typedef long long ll;

struct Builder {
    vector<ll> h, c;
    vector<array<ll,3>> edges; // a,b,w ; -1 marks the sink
    ll cheapCost = 0;          // odometer cost of L + chain2 piles only (sizes K)

    int addV(ll hh, ll cc) { h.push_back(hh); c.push_back(cc); return (int)h.size() - 1; }
    void edge(int a, int b, ll w) { edges.push_back({(ll)a, (ll)b, w}); }

    // flavor H: small degree, huge height -> huge raw height AND huge cost
    void addH(ll wlo, ll whi, ll heightLo, ll heightHi, ll clo, ll chi) {
        ll w = rnd.next(wlo, whi);
        ll height = rnd.next(heightLo, heightHi);
        ll crit = rnd.next(clo, chi);
        int v = addV(height, crit);
        edge(v, -1, w);
        // deliberately NOT added to cheapCost: H must stay far outside budget
    }
    // flavor L: deg large, height = w*mult + jitter (jitter<w) -> cost EXACTLY mult
    void addL(ll wlo, ll whi, ll multLo, ll multHi, ll clo, ll chi) {
        ll w = rnd.next(wlo, whi);
        ll mult = rnd.next(multLo, multHi);
        ll jitter = (w > 1) ? rnd.next(0LL, w - 1) : 0;
        ll height = w * mult + jitter;
        ll crit = rnd.next(clo, chi);
        int v = addV(height, crit);
        edge(v, -1, w);
        cheapCost += mult;
    }
    // small cascading pair: v1 (sink w0, v2 w1), v2 (only v1, w1). Both "cheap".
    void addChain2(ll wlo, ll whi, ll multLo, ll multHi, ll clo, ll chi) {
        ll w0 = rnd.next(wlo, whi);
        ll w1 = rnd.next(wlo, whi);
        ll deg1 = w0 + w1;
        ll mult1 = rnd.next(multLo, multHi);
        ll jitter1 = rnd.next(0LL, deg1 - 1);
        ll h1 = deg1 * mult1 + jitter1;
        ll mult2 = rnd.next(multLo, multHi);
        ll jitter2 = (w1 > 1) ? rnd.next(0LL, w1 - 1) : 0;
        ll h2 = w1 * mult2 + jitter2;
        int v1 = addV(h1, rnd.next(clo, chi));
        int v2 = addV(h2, rnd.next(clo, chi));
        edge(v1, -1, w0);
        edge(v1, v2, w1);
        // safe over-estimate of the cost to fully stabilize the pair
        cheapCost += mult1 + mult2 + 2;
    }
};

static void printAll(Builder& B, ll K) {
    int n = (int)B.h.size();
    int m = (int)B.edges.size();
    printf("%d %d %lld\n", n, m, K);
    for (auto& e : B.edges) {
        ll a = e[0], b = e[1], w = e[2];
        if (a == -1) a = n;
        if (b == -1) b = n;
        printf("%lld %lld %lld\n", a, b, w);
    }
    for (int i = 0; i < n; i++) printf("%lld%c", B.h[i], i + 1 == n ? '\n' : ' ');
    for (int i = 0; i < n; i++) printf("%lld%c", B.c[i], i + 1 == n ? '\n' : ' ');
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    Builder B;
    ll slack = 5; // extra budget headroom beyond "fund every cheap thing"

    if (testId == 1) {
        // tiny sanity/example-scale case: no hub trap yet, partial budget
        for (int i = 0; i < 5; i++) B.addL(4, 12, 1, 3, 2, 10);
        ll K = max(4LL, (ll)(B.cheapCost * 0.55));
        printAll(B, K);
        return 0;
    } else if (testId == 2) {
        // small normal mix, partial budget, one cascading pair
        for (int i = 0; i < 8; i++) B.addL(6, 20, 1, 3, 3, 15);
        B.addChain2(6, 14, 1, 2, 4, 10);
        ll K = max(5LL, (ll)(B.cheapCost * 0.55));
        printAll(B, K);
        return 0;
    } else if (testId == 3) {
        // no extreme trap yet: a mildly graded mix, budget = ~55% of full cost
        for (int i = 0; i < 10; i++) B.addL(6, 18, 1, 3, 2, 10);
        for (int i = 0; i < 3; i++) B.addChain2(6, 14, 1, 3, 3, 9);
        ll K = max(6LL, (ll)(B.cheapCost * 0.55));
        printAll(B, K);
        return 0;
    } else if (testId == 4) {
        // TRAP #1: one loud, hopeless hub among many cheap piles
        B.addH(2, 3, 5000, 9000, 1, 3);
        for (int i = 0; i < 20; i++) B.addL(30, 55, 1, 3, 10, 25);
        slack = 5;
    } else if (testId == 5) {
        for (int i = 0; i < 2; i++) B.addH(2, 3, 5000, 9000, 1, 3);
        for (int i = 0; i < 30; i++) B.addL(30, 55, 1, 4, 10, 25);
        for (int i = 0; i < 3; i++) B.addChain2(20, 45, 1, 3, 8, 20);
        slack = 8;
    } else if (testId == 6) {
        // TRAP: several medium-loud hubs instead of one giant one
        for (int i = 0; i < 5; i++) B.addH(3, 6, 2000, 4000, 1, 4);
        for (int i = 0; i < 60; i++) B.addL(25, 55, 1, 4, 10, 25);
        slack = 10;
    } else if (testId == 7) {
        for (int i = 0; i < 3; i++) B.addH(2, 4, 7000, 12000, 1, 4);
        for (int i = 0; i < 100; i++) B.addL(25, 55, 1, 4, 10, 25);
        for (int i = 0; i < 10; i++) B.addChain2(20, 45, 1, 3, 8, 20);
        slack = 15;
    } else if (testId == 8) {
        for (int i = 0; i < 10; i++) B.addH(2, 5, 4000, 9000, 1, 4);
        for (int i = 0; i < 300; i++) B.addL(20, 55, 1, 4, 8, 25);
        for (int i = 0; i < 30; i++) B.addChain2(15, 45, 1, 3, 6, 20);
        slack = 25;
    } else if (testId == 9) {
        for (int i = 0; i < 20; i++) B.addH(2, 5, 5000, 12000, 1, 5);
        for (int i = 0; i < 600; i++) B.addL(20, 55, 1, 4, 8, 25);
        for (int i = 0; i < 60; i++) B.addChain2(15, 45, 1, 3, 6, 20);
        slack = 40;
    } else { // testId == 10 : fill the size envelope (n up to 2000)
        for (int i = 0; i < 30; i++) B.addH(2, 5, 6000, 15000, 1, 5);
        for (int i = 0; i < 1770; i++) B.addL(20, 55, 1, 4, 8, 25);
        for (int i = 0; i < 100; i++) B.addChain2(15, 45, 1, 3, 6, 20);
        slack = 60;
    }

    ll K = B.cheapCost + slack;
    printAll(B, K);
    return 0;
}
