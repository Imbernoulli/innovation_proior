#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

// -----------------------------------------------------------------------------
// Generator for "Fragment-Based Ligand Assembly".
//
// Emits a fragment library whose reward landscape is deliberately adversarial:
//   * DECOY  : one high-b, light fragment -> defines the single-fragment baseline B.
//   * TRAP   : one very-high-b but over-budget (w >> Wmax) fragment. A "grab the
//              biggest b" heuristic includes it and gets crushed by the weight
//              penalty; the intended optimum leaves it out.
//   * PLANTED synergy cluster: a group of same-class (SYN=0) fragments with large
//              q[0][0] and valence 3-4, so a ring-rich core scores highly. Buried
//              among noise (indices shuffled).
//   * NEEDLE : a small set of class-1 / class-2 fragments whose CROSS bond q[1][2]
//              is very large but whose own b is small -> an affinity-first greedy
//              overlooks them; only class-aware search captures the synergy.
//   * NOISE  : heavier random fragments; including many overflows the weight budget.
//
// Output:  M C Wmax Lam Rb  then M lines  w b val p  then C lines of C ints (q).
// -----------------------------------------------------------------------------

struct Frag { ll w, b, val, p; };

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    double f = (testId - 1) / 9.0;

    int C    = 3 + (int)llround(f * 3.0);              // 3..6 pharmacophore classes
    int M    = 30 + (int)llround(f * 1970.0);          // 30..2000 fragments (haystack ladder)
    ll  Lam  = 24 + (ll)llround(f * 14.0);             // 24..38 penalty coefficient
    ll  Rb   = 18;                                     // ring bonus (constant reward)
    // Wmax set below relative to the planted-structure weight so the budget BINDS.

    // ---- interaction matrix q (symmetric) ----
    vector<vector<ll>> q(C, vector<ll>(C, 0));
    for (int a = 0; a < C; a++)
        for (int c = a; c < C; c++){
            ll v;
            if (a == c) v = -(15 + rnd.next(0, 25));       // most same-class bonds clash mildly
            else        v = -(5 + rnd.next(0, 25));        // cross bonds clash mildly
            q[a][c] = q[c][a] = v;
        }
    ll QSYN    = 22;                                    // synergy cluster (class 0), constant
    ll QNEEDLE = 55;                                    // needle cross (classes 1,2), constant
    q[0][0] = QSYN;
    q[1][2] = q[2][1] = QNEEDLE;
    q[1][1] = -(30 + rnd.next(0, 20));                 // needle atoms clash with own class
    q[2][2] = -(30 + rnd.next(0, 20));

    // ---- assemble fragment list ----
    vector<Frag> F;
    ll plantedWeight = 0;                              // running weight of the good structure

    // DECOY: sets baseline B (light, best single b) -- constant so B and headroom stable
    ll bDecoy = 260;
    ll wDecoy = 60 + rnd.next(0, 20);
    F.push_back({ wDecoy, bDecoy, 2, (ll)(C - 1) });
    plantedWeight += wDecoy;

    // PLANTED synergy cluster (class 0) -- CONSTANT size (reward independent of test size)
    int nc = 8;
    for (int i = 0; i < nc; i++){
        ll ww = 35 + rnd.next(0, 15);
        F.push_back({ ww, 16 + rnd.next(0, 16), 3, 0 });
        plantedWeight += ww;
    }

    // NEEDLE fragments: alternating classes 1 and 2, small b, valence 2-3 -- CONSTANT count
    int np = 4;                                        // pairs
    for (int i = 0; i < 2 * np; i++){
        ll cls = (i % 2 == 0) ? 1 : 2;
        ll ww = 40 + rnd.next(0, 15);
        F.push_back({ ww, 12 + rnd.next(0, 12), 2 + rnd.next(0, 1), cls });
        plantedWeight += ww;
    }

    // Weight budget BINDS: only ~65% of the good structure fits for free, so the best
    // ligand must trade fragment/bond value against the Lipinski overflow penalty.
    ll Wmax = (ll)llround(0.55 * (double)plantedWeight) + 30 + rnd.next(0, 30);

    // TRAP: huge b but far over budget (single-fragment value strongly negative)
    F.push_back({ Wmax + 500 + rnd.next(0, 500), 290 + rnd.next(0, 10),
                  1, (ll)rnd.next(0, C - 1) });

    // NOISE fill up to M: mostly net-negative decoy fragments (heavy, low/negative b,
    // clashing bonds) so that the optimal ligand is the FIXED-SIZE planted structure,
    // not "add every fragment". This keeps the reward bounded relative to B on large
    // instances (headroom) and forces search to LOCATE the good substructure.
    while ((int)F.size() < M){
        ll cls = rnd.next(0, C - 1);
        ll bb  = rnd.next(-150, 5);                     // rarely worth adding
        F.push_back({ 40 + rnd.next(0, 200), bb, 1 + rnd.next(0, 3), cls });
    }
    // (if base list already exceeds M, trim noise-free — but base <= M by construction)
    while ((int)F.size() > M) F.pop_back();

    // ---- shuffle so planted structure is not positional ----
    for (int i = (int)F.size() - 1; i > 0; i--) swap(F[i], F[rnd.next(0, i)]);

    // ---- emit ----
    printf("%d %d %lld %lld %lld\n", M, C, Wmax, Lam, Rb);
    for (auto &fr : F)
        printf("%lld %lld %lld %lld\n", fr.w, fr.b, fr.val, fr.p);
    for (int a = 0; a < C; a++){
        for (int c = 0; c < C; c++)
            printf("%lld%c", q[a][c], c + 1 == C ? '\n' : ' ');
    }
    return 0;
}
