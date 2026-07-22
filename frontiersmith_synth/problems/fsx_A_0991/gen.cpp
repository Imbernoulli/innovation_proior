#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// "Bolt and Braid: One Shop's Cutting Cards"   family: field-braid-partitioner
//
// F measurement components, each with a card-width w_f. Q fixed pattern orders,
// each needing a small subset of components. Solver assigns each component a
// HOME card and optionally a BRAID (second, different) card, subject to a global
// replication budget R. An order's cost is K per DISTINCT card in its cheapest
// cover plus the batch-scaled footprint of those cards (choosing, per
// component, whichever of its <=2 hosts is cheaper). The flat per-card fee K
// is what makes touching fewer, well-chosen cards pay off -- without it,
// printing every component on its own card is always optimal.
//
// PLANTED STRUCTURE (hidden from the checker, which only re-derives a one-card
// baseline): the instance is `numPairs` independent "pair gadgets". Gadget k
// has clusterA_k and clusterB_k, each split into several small MINI-CLIQUES
// (fixed size 4) of fields that recur TOGETHER, order after order -- real
// co-access locality, not a one-off random subset -- so the right card for
// each mini-clique is exactly that mini-clique (any bigger merge wastes
// footprint on fields that clique's orders never need; any smaller split pays
// extra pull fees for no reason). On top of that:
//   - X_k: one NARROW field queried heavily alongside BOTH sides' first
//     mini-clique (many high-row "AX" and "BX" orders). Whichever side X_k's
//     card ends up hosting it, the OTHER side's orders now need two cards for
//     one order line. Only REPLICATING X_k (cheap, since it's narrow) collapses
//     those covers back to one card -- a strict partition can never do this
//     since X_k must live in exactly one place.
//   - Y_k: one WIDE field with only a handful of LOW-row orders alongside the
//     first A mini-clique -- enough raw co-occurrence COUNT to look attractive
//     to a frequency-only affinity miner, but low actual traffic weight, so
//     bolting it onto that clique's card inflates the footprint every one of
//     that clique's (much more frequent) orders pays for. It should be
//     quarantined onto its own card instead, even though it is "friends" with
//     that clique.
//
// Background fields/orders add unrelated noise (their own tiny disjoint
// co-access components). Schema breadth (numPairs, mini-clique counts) grows
// only mildly across tests -- the one-card baseline's advantage grows
// quadratically with schema breadth (it charges the WHOLE schema's width to
// every order, and both schema width and order count grow with breadth), so
// size instead scales mainly through repeated ORDER VOLUME on the same
// structure, which grows the baseline and any good solution's cost in the
// same proportion.
//
// Replication budget R == numPairs: exactly enough to braid every X_k once.
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    // numPairs and numMiniA are held FIXED across every test: the one-card
    // baseline's advantage over any good solution grows with SCHEMA BREADTH
    // (it scales roughly with numPairs*numMiniA, since baseline charges the
    // whole schema's width to every order while a good solution's per-order
    // cost stays bounded to one mini-clique's footprint) -- growing it at all
    // saturates the score past the cap within a couple of tests. Difficulty
    // and input size instead scale entirely through repeated ORDER VOLUME on
    // the SAME structure, which grows the baseline and any good solution's
    // cost in the same proportion (constant ratio) while still filling out
    // the size envelope.
    int numPairs = 2;
    int numMiniA = 3;
    int numMiniB = numMiniA;
    const int MINI = 4;                                       // fixed mini-clique size
    int nBG = 8 + (int)llround(f * 12.0);                      // 8..20, small, additive-only
    int queriesPerClique = 8 + (int)llround(f * 225.0);        // 8..233 (drives Q growth)

    vector<ll> w;
    w.push_back(0);
    struct Qy { ll rows; vector<int> ids; };
    vector<Qy> qs;

    // Aclq[k][c] / Bclq[k][c] = field ids of mini-clique c of pair k's side
    vector<vector<vector<int>>> Aclq(numPairs+1), Bclq(numPairs+1);
    vector<int> Xid(numPairs+1), Yid(numPairs+1);

    for (int k = 1; k <= numPairs; k++){
        Aclq[k].resize(numMiniA);
        for (int c = 0; c < numMiniA; c++)
            for (int i = 0; i < MINI; i++){
                w.push_back(15 + rnd.next(0, 25));            // 15..40
                Aclq[k][c].push_back((int)w.size() - 1);
            }
        Bclq[k].resize(numMiniB);
        for (int c = 0; c < numMiniB; c++)
            for (int i = 0; i < MINI; i++){
                w.push_back(15 + rnd.next(0, 25));            // 15..40
                Bclq[k][c].push_back((int)w.size() - 1);
            }
        w.push_back(2 + rnd.next(0, 4));                      // X: 2..6 narrow
        Xid[k] = (int)w.size() - 1;
        w.push_back(250 + rnd.next(0, 150));                  // Y: 250..400 wide
        Yid[k] = (int)w.size() - 1;
    }
    vector<int> BGids;
    for (int i = 0; i < nBG; i++){
        w.push_back(10 + rnd.next(0, 70));                    // 10..80
        BGids.push_back((int)w.size() - 1);
    }

    int F = (int)w.size() - 1;

    auto pickSubset = [&](vector<int> pool, int lo, int hi) -> vector<int> {
        int k = min((int)pool.size(), lo + rnd.next(0, max(0, hi - lo)));
        for (int i = (int)pool.size() - 1; i > 0; i--) swap(pool[i], pool[rnd.next(0, i)]);
        pool.resize(k);
        return pool;
    };

    for (int k = 1; k <= numPairs; k++){
        // clique-local orders: EVERY order stays within one mini-clique --
        // real recurring co-access locality, not a fresh random subset.
        for (int c = 0; c < numMiniA; c++)
            for (int r = 0; r < queriesPerClique; r++){
                vector<int> ids = pickSubset(Aclq[k][c], 2, MINI);
                if (ids.size() < 2) continue;
                qs.push_back({500 + rnd.next(0, 1000), ids});
            }
        for (int c = 0; c < numMiniB; c++)
            for (int r = 0; r < queriesPerClique; r++){
                vector<int> ids = pickSubset(Bclq[k][c], 2, MINI);
                if (ids.size() < 2) continue;
                qs.push_back({500 + rnd.next(0, 1000), ids});
            }
        // AX / BX bridge orders (HIGH ROW WEIGHT per order -- real monetary
        // impact -- but a MODEST order COUNT, split across the 4 clique
        // fields: each individual (field,X) pairwise co-occurrence count must
        // stay BELOW a same-clique pair's own count (~0.56*queriesPerClique,
        // since a random 2..4 subset of 4 fields contains any fixed pair with
        // mean probability ~0.56), or a raw-count affinity miner would let X
        // rip the clique apart / bridge both sides before the clique's own
        // cohesion edges are even processed. Kept well below that here.
        int nAX = max(6, (int)llround(0.45 * queriesPerClique));
        for (int r = 0; r < nAX; r++){
            int a = Aclq[k][0][rnd.next(0, MINI - 1)];
            qs.push_back({800 + rnd.next(0, 1200), {a, Xid[k]}});
        }
        int nBX = nAX;
        for (int r = 0; r < nBX; r++){
            int b = Bclq[k][0][rnd.next(0, MINI - 1)];
            qs.push_back({800 + rnd.next(0, 1200), {b, Xid[k]}});
        }
        // AY decoy orders: nonzero co-occurrence COUNT (enough that a
        // raw-count affinity miner still tries to merge it in once the
        // clique + X have settled), but LOW row weight, and weaker in COUNT
        // than the X bridge above so it doesn't out-rank it.
        int nAY = max(4, (int)llround(0.18 * queriesPerClique));
        for (int r = 0; r < nAY; r++){
            int a = Aclq[k][0][rnd.next(0, MINI - 1)];
            qs.push_back({50 + rnd.next(0, 100), {a, Yid[k]}});
        }
    }
    // background noise orders (own disjoint co-access components)
    for (int rep = 0; rep < 3; rep++)
        for (int i = 0; i + 1 < (int)BGids.size(); i++){
            vector<int> ids = pickSubset(BGids, 2, 5);
            if (ids.size() < 2) continue;
            qs.push_back({300 + rnd.next(0, 900), ids});
        }

    int R = numPairs;
    ll P = 4000;
    ll K = 50;    // flat per-card pull fee

    // shuffle order list so structure isn't given away by input order
    for (int i = (int)qs.size() - 1; i > 0; i--) swap(qs[i], qs[rnd.next(0, i)]);

    int Q = (int)qs.size();
    printf("%d %d %lld %lld %d\n", F, Q, P, K, R);
    for (int i = 1; i <= F; i++) printf("%lld\n", w[i]);
    for (auto &q : qs){
        printf("%lld %d", q.rows, (int)q.ids.size());
        for (int id : q.ids) printf(" %d", id);
        printf("\n");
    }
    return 0;
}
