#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Quarry Ignition generator (family: chain-detonation-closure)
//
// Charges are assembled from independent, spatially separated COMPONENTS so the
// weakly-connected-component structure is real and exploitable:
//
//   * Y-STRUCTURE (the planted insight): a shielded target charge (c=0, big value,
//     threshold = number of branches K) fed by K axis-aligned chains of length
//     `branchLen`. Only the outermost node of each chain is ignitable; consecutive
//     nodes are exactly `D` apart with radius exactly `D`, so each chain is a clean
//     multi-hop relay into the target. The target only detonates once ALL K chains
//     are lit -- a genuine coordination requirement, not a smooth marginal gain.
//
//   * DENSE CLUSTER (the trap): many ignitable charges packed into a small box with
//     radius large enough that they all mutually reach each other (threshold 1, so
//     ONE spark floods it all). Every candidate in it reports a huge one-shot
//     payoff, which is exactly what lures a single continuous marginal-gain pass
//     into treating it as unbeatable -- while a shielded Y-structure target, whose
//     payoff only appears once BOTH its feeder chains are lit together, never looks
//     attractive round-by-round and is left dark.
//
//   * DECOYS (the other trap): isolated (r=0) ignitable charges with a large own
//     value. They lure a value-only baseline but detonate nothing beyond themselves.
//
// Components are placed in fixed-width slots far enough apart that no cross-
// component edges can ever form (radius is always << slot separation), so the
// intended weakly-connected-component decomposition is exact and verifiable.
// -----------------------------------------------------------------------------

static vector<ll> X, Y, V, R, T;
static vector<int> C;
static ll slotX = 0;
static const ll SLOT_W = 1300;

static int add(ll x, ll y, ll v, ll r, ll t, int c){
    X.push_back(x); Y.push_back(y); V.push_back(v); R.push_back(r); T.push_back(t); C.push_back(c);
    return (int)X.size() - 1;
}

// K axis-aligned branches (K in 2..4), each of length branchLen, feeding a shielded
// target of value targetV and threshold K. Consecutive link value linkV (randomized
// a little per node so components aren't perfectly uniform).
static void buildY(int K, int branchLen, ll D, ll linkVlo, ll linkVhi, ll targetV){
    ll cx = slotX + 700, cy = 500;
    add(cx, cy, targetV, 0, K, 0);
    static const ll dxs[4] = {-1, 1, 0, 0};
    static const ll dys[4] = {0, 0, -1, 1};
    for (int side = 0; side < K; side++){
        ll dx = dxs[side], dy = dys[side];
        for (int k = 0; k < branchLen; k++){
            ll nx = cx + dx * D * (k + 1), ny = cy + dy * D * (k + 1);
            int isEntry = (k == branchLen - 1);
            ll v = rnd.next((int)linkVlo, (int)linkVhi);
            add(nx, ny, v, D, 1, isEntry ? 1 : 0);
        }
    }
    slotX += SLOT_W;
}

// A tight clique-like cluster: `count` ignitable charges in a `box`x`box` square
// with radius large enough that they all mutually reach each other (threshold 1).
// A SINGLE well-chosen ignition here floods the whole cluster -- that is
// deliberate: it is what makes the cluster look so attractive one hop out (every
// candidate in it reports a huge immediate/one-shot payoff), tempting a
// value-blind or single-pass heuristic to over-invest multiple picks in a
// component that only ever needed one.
static void buildCluster(int count, ll valLo, ll valHi, ll box){
    ll ox = slotX + 50, oy = 500;
    ll radius = box + box / 2;
    for (int i = 0; i < count; i++){
        ll x = ox + rnd.next(0, (int)box);
        ll y = oy + rnd.next(0, (int)box);
        ll v = rnd.next((int)valLo, (int)valHi);
        add(x, y, v, radius, 1, 1);
    }
    slotX += SLOT_W;
}

// Isolated, high individual value, zero reach: bait for a value-only baseline.
static void buildDecoys(int count, ll valLo, ll valHi){
    ll ox = slotX, oy = 500;
    for (int i = 0; i < count; i++){
        ll x = ox + (ll)i * 25, y = oy;
        ll v = rnd.next((int)valLo, (int)valHi);
        add(x, y, v, 0, 1, 1);
    }
    slotX += SLOT_W;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    X.clear(); Y.clear(); V.clear(); R.clear(); T.clear(); C.clear(); slotX = 0;

    int M = 2;

    if (testId == 1){
        // tiny sanity case, mirrors the statement's worked example in shape.
        buildY(2, 1, 6, 3, 3, 100);
        buildDecoys(2, 40, 55);
        M = 2;
    } else if (testId == 2){
        // small, mostly generic: no planted trap yet, just baseline realism.
        buildCluster(34, 4, 12, 60);
        buildDecoys(3, 30, 60);
        M = 3;
    } else if (testId == 3){
        buildY(2, 1, 8, 3, 5, 150);
        buildCluster(30, 4, 12, 60);
        buildDecoys(4, 35, 65);
        M = 4;
    } else if (testId == 4){
        // TRAP #1: one 2-branch structure vs. a denser cluster + pricier decoys.
        // Decoy count matches M so the value-only baseline never needs to dip into
        // the cluster/entries itself (which would let it "accidentally" flood the
        // cluster too and erase the intended gap).
        buildY(2, 2, 10, 2, 5, 300);
        buildCluster(42, 4, 14, 70);
        buildDecoys(5, 45, 90);
        M = 5;
    } else if (testId == 5){
        // TRAP #2: two structures competing with cluster+decoys for the budget.
        buildY(2, 2, 10, 2, 5, 400);
        buildY(2, 2, 12, 2, 5, 350);
        buildCluster(48, 4, 14, 80);
        buildDecoys(6, 55, 110);
        M = 6;
    } else if (testId == 6){
        // TRAP #3: deeper 3-hop chains (tests true multi-hop closure reasoning).
        buildY(2, 3, 12, 2, 4, 600);
        buildY(2, 2, 10, 2, 4, 500);
        buildCluster(58, 5, 16, 90);
        buildDecoys(8, 65, 140);
        M = 8;
    } else if (testId == 7){
        // TRAP #4: three structures, all affordable -- tests broad discovery.
        buildY(2, 2, 10, 2, 4, 700);
        buildY(2, 2, 11, 2, 4, 550);
        buildY(2, 3, 9, 2, 4, 450);
        buildCluster(64, 5, 16, 90);
        buildDecoys(10, 65, 140);
        M = 10;
    } else if (testId == 8){
        // TRAP + NEEDLE: a 3-branch structure (threshold 3) hides a huge target
        // among two ordinary structures, a big cluster, and many decoys. Budget is
        // tight enough that missing the needle costs dearly.
        buildY(3, 3, 10, 2, 4, 3000);
        buildY(2, 2, 10, 2, 4, 500);
        buildY(2, 2, 9, 2, 4, 400);
        buildCluster(70, 5, 16, 90);
        buildDecoys(10, 65, 150);
        M = 10;
    } else if (testId == 9){
        // Genuine knapsack across four structures: budget covers only 3 of them,
        // so the low-value one must be sacrificed.
        buildY(2, 2, 10, 2, 4, 300);
        buildY(2, 2, 11, 2, 4, 500);
        buildY(2, 3, 9, 2, 4, 700);
        buildY(2, 2, 12, 2, 4, 900);
        buildCluster(64, 5, 16, 90);
        buildDecoys(6, 65, 140);
        M = 6;
    } else {
        // testId 10: largest scale, fills the constraint envelope. Six structures
        // (mixed K=2/3, so costs are 2 or 3) with total cost 15 but M=14, forcing a
        // real value/cost trade-off, plus a big cluster and many decoys.
        buildY(2, 2, 10, 2, 4, 800);          // cost 2
        buildY(3, 3, 9, 2, 4, 1500);          // cost 3
        buildY(2, 2, 11, 2, 4, 600);          // cost 2
        buildY(3, 3, 10, 2, 4, 2200);         // cost 3
        buildY(2, 2, 12, 2, 4, 1000);         // cost 2
        buildY(3, 3, 11, 2, 4, 2500);         // cost 3
        buildCluster(150, 5, 18, 130);
        buildDecoys(14, 70, 160);
        M = 14;
    }

    int N = (int)X.size();
    // Ensure enough ignitable charges exist for M (generator invariant).
    int nIg = 0; for (int c : C) nIg += c;
    if (nIg < M){
        // extremely defensive fallback; not expected to trigger given the plan above.
        buildDecoys(M - nIg + 1, 10, 20);
        N = (int)X.size();
    }

    printf("%d %d\n", N, M);
    for (int i = 0; i < N; i++)
        printf("%lld %lld %lld %lld %lld %d\n", X[i], Y[i], V[i], R[i], T[i], C[i]);
    return 0;
}
