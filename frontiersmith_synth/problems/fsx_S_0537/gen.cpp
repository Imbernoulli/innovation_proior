#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ----------------------------------------------------------------------------
// braess-toll-surgeon : city road tolls before a festival weekend.
// The network is a disjoint union of Braess "district" gadgets (4 nodes each):
//     S -> A (congestion), A -> T (fixed), S -> B (fixed), B -> T (congestion),
//     A -> B (the paradox shortcut, latency 0).
// A PARADOX gadget: everyone floods S->A->B->T at the zero-toll equilibrium and
//   deleting the shortcut (a prohibitive toll on A->B) strictly lowers travel time.
// A DECOY gadget: the "shortcut-shaped" road A->B is genuinely load-bearing;
//   deleting it makes travel time WORSE.  (Congestion pricing cannot tell them
//   apart, and it never taxes the zero-latency shortcut anyway.)
// testId is a difficulty/structure ladder; several tests are trap-heavy (almost
// all paradox gadgets) so smooth congestion-proportional pricing lands far below
// the surgical optimum.
// ----------------------------------------------------------------------------

struct E { int u, v; long long a; int k; long long b; };

static long long ipow(long long x, int k){
    long long r = 1;
    for (int i = 0; i < k; i++){ r *= x; if (r > (long long)4e15){ r = (long long)4e15; break; } }
    return r;
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // ---- number of districts (gadgets) ----
    int G = (t == 1) ? 1 : min(34, 2 + 3 * t);

    // ---- probability a district is a paradox (vs decoy), per test ----
    //  trap-heavy tests {4,6,7,9,10}: almost all paradox.
    //  selection tests {3,5,8}: ~half decoys, so "delete every shortcut" is wrong.
    int par_pct;
    if (t == 1) par_pct = 100;
    else if (t == 2) par_pct = 55;
    else if (t == 3 || t == 5 || t == 8) par_pct = 42;
    else par_pct = 68;

    // ---- budget tightness rho (fraction of total deletion cost affordable) ----
    double rho;
    if (t == 1) rho = 1.00;
    else if (t == 2) rho = 0.90;
    else if (t == 3 || t == 5 || t == 8) rho = 0.70;
    else rho = 0.85;

    vector<E> edges;
    vector<pair<int,int>> comm;   // (s,t) 0-indexed
    long long delete_cost_sum = 0; // sum over paradox gadgets of the toll needed to delete its shortcut

    int node = 0;
    for (int g = 0; g < G; g++){
        int S = node + 0, A = node + 1, B = node + 2, T = node + 3;
        node += 4;

        // demand: even so the split is clean
        int D = (t == 1) ? 6 : (rnd.next(4, 8) * 2); // 8..16
        // congestion degree (paradox strength). tiny example uses quadratic.
        int p = (t == 1) ? 2 : rnd.next(3, 4);

        bool paradox = (rnd.next(1, 100) <= par_pct);

        if (paradox){
            long long beta = ipow(D, p) + 1;      // strict Braess threshold
            // S->A congestion x^p ; A->T fixed beta ; S->B fixed beta ; B->T congestion x^p
            edges.push_back({S, A, 1, p, 0});
            edges.push_back({A, T, 0, 1, beta});
            edges.push_back({S, B, 0, 1, beta});
            edges.push_back({B, T, 1, p, 0});
            edges.push_back({A, B, 0, 1, 0});      // paradox shortcut, latency 0
            delete_cost_sum += beta;               // toll needed to delete this shortcut
        } else {
            // DECOY: fixed detours are very expensive, so the shortcut A->B is
            // genuinely useful; deleting it is harmful.  The congested edges carry
            // a high congestion coefficient, so a naive "toll roads in proportion
            // to their congestion sensitivity" rule dumps budget here and HURTS
            // (it pushes flow off the good shortcut onto the ruinous detours).
            long long BIG = 2 * ipow(D, p);        // large fixed cost on the detours
            edges.push_back({S, A, 3, 1, 0});
            edges.push_back({A, T, 0, 1, BIG});
            edges.push_back({S, B, 0, 1, BIG});
            edges.push_back({B, T, 3, 1, 0});
            edges.push_back({A, B, 0, 1, 0});      // load-bearing "shortcut-shaped" road
        }
        for (int i = 0; i < D; i++) comm.push_back({S, T});
    }

    int N = node;
    int M = (int)edges.size();
    int Dtot = (int)comm.size();
    int R = 40;
    long long Tbudget = (long long)llround(rho * (double)delete_cost_sum);
    if (Tbudget < 0) Tbudget = 0;

    printf("%d %d %d %lld %d\n", N, M, Dtot, Tbudget, R);
    for (auto &e : edges)
        printf("%d %d %lld %d %lld\n", e.u + 1, e.v + 1, e.a, e.k, e.b);
    for (auto &c : comm)
        printf("%d %d\n", c.first + 1, c.second + 1);
    return 0;
}
