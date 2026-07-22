#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Generator for "Day-Night Courier Laplacians".
// Builds a logical graph (node ids 0..n-1) as a union of:
//   T1 = two "hub" stars (center 0 <-> leaves 1..k-1 ;  center k <-> leaves k+1..2k-1) plus one
//        bridge edge (k-1, k).           [n1 = 2k nodes, |T1| = 2k-1 edges, weight 1]
//   T2 = a perfect matching (i, k+i) for i=0..k-1 across the two halves.
//                                        [|T2| = k edges, weight 1]
//   noise = random extra edges, weight in [1,2].
// T1's two stars are excellent LOCAL diffusers (min nonzero Laplacian eigenvalue = 1) but their
// only path across the L/R halves is the single bridge -> T1 alone has one very slow mode (the
// L-vs-R imbalance). T2's matching does nothing local (huge kernel) but crushes exactly that
// L-vs-R imbalance (eigenvalue 2 for every matched pair's antisymmetric mode). Used as ALTERNATE
// day/night networks (T1 day, T2 night) the two weaknesses cancel; mixed 50/50 they don't.
//
// Finally: RELABEL nodes with a random permutation and shuffle the edge print order, so no
// positional/index heuristic leaks the planted structure.

struct E { int u, v, w; int role; };

static int NN; // logical node counter
static vector<E> edges;
#ifdef DEBUG_ROLES
static bool debugRoles = true;
#else
static bool debugRoles = false;
#endif

static void addEdge(int u, int v, int w, int role = 2) { edges.push_back({u, v, w, role}); }

// Appends the star+bridge+matching trap on a FRESH block of 2k logical nodes starting at base.
// Returns (t1count, t2count). role 0 = T1 (star+bridge), role 1 = T2 (matching).
static pair<int,int> addTrap(int base, int k) {
    int t1 = 0, t2 = 0;
    int cL = base, cR = base + k;
    for (int i = 1; i < k; i++) { addEdge(cL, base + i, 1, 0); t1++; }
    for (int i = 1; i < k; i++) { addEdge(cR, base + k + i, 1, 0); t1++; }
    addEdge(base + k - 1, base + k, 1, 0); t1++;             // bridge
    for (int i = 0; i < k; i++) { addEdge(base + i, base + k + i, 1, 1); t2++; }
    return {t1, t2};
}

static void addNoise(int n, int lo, int hi, int cnt) {
    for (int c = 0; c < cnt; c++) {
        int u = rnd.next(0, n - 1), v = rnd.next(0, n - 1);
        while (v == u) v = rnd.next(0, n - 1);
        addEdge(u, v, rnd.next(lo, hi), 2);
    }
}

// generic (non-trap) connected random graph on n nodes, ~targetM edges.
static void addGeneric(int n, int targetM, int wlo, int whi) {
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0);
    for (int i = n - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);
    for (int i = 0; i + 1 < n; i++) addEdge(perm[i], perm[i + 1], rnd.next(wlo, whi));
    int have = n - 1;
    set<pair<int,int>> seen;
    for (int i = 0; i + 1 < n; i++) seen.insert({min(perm[i], perm[i+1]), max(perm[i], perm[i+1])});
    int guard = 0;
    while (have < targetM && guard < targetM * 30 + 1000) {
        guard++;
        int u = rnd.next(0, n - 1), v = rnd.next(0, n - 1);
        if (u == v) continue;
        auto key = make_pair(min(u, v), max(u, v));
        if (seen.count(key)) continue;
        seen.insert(key);
        addEdge(u, v, rnd.next(wlo, whi));
        have++;
    }
}

static void emit(int n, double tau, long long capDay, long long capNight) {
    int m = (int)edges.size();
    // random relabel
    vector<int> perm(n);
    iota(perm.begin(), perm.end(), 0);
    for (int i = n - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);
    for (auto& e : edges) { e.u = perm[e.u]; e.v = perm[e.v]; }
    // shuffle print order
    for (int i = m - 1; i > 0; i--) swap(edges[i], edges[rnd.next(0, i)]);

    printf("%d %d\n", n, m);
    printf("%.6f\n", tau);
    printf("%lld %lld\n", capDay, capNight);
    for (auto& e : edges) printf("%d %d %d\n", e.u + 1, e.v + 1, e.w);
    if (debugRoles) {
        fprintf(stderr, "ROLES");
        for (auto& e : edges) fprintf(stderr, " %d", e.role);
        fprintf(stderr, "\n");
    }
}

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    edges.clear();

    if (testId == 1) {
        // tiny sanity
        int k = 3;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 1, 1);
        int m = (int)edges.size();
        long long capNight = cnt.second + 1;
        long long capDay = m - capNight + 1;
        emit(n, 2.0, capDay, capNight);
    } else if (testId == 2) {
        // generic random graph, no trap: ladder sanity case
        int n = 18;
        addGeneric(n, 30, 1, 3);
        int m = (int)edges.size();
        emit(n, 1.2, m / 2 + 2, m - m / 2 + 2);
    } else if (testId == 3) {
        int k = 6;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 4);
        int m = (int)edges.size();
        long long capNight = cnt.second + 2 + 3;
        long long capDay = cnt.first + 2 + 3;
        emit(n, 2.8, capDay, capNight);
    } else if (testId == 4) {
        int k = 9;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 6);
        int m = (int)edges.size();
        long long capNight = cnt.second + 3 + 3;
        long long capDay = cnt.first + 3 + 3;
        emit(n, 2.9, capDay, capNight);
    } else if (testId == 5) {
        int k = 12;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 10);
        int m = (int)edges.size();
        long long capNight = cnt.second + 5 + 3;
        long long capDay = cnt.first + 5 + 3;
        emit(n, 3.0, capDay, capNight);
    } else if (testId == 6) {
        // NEEDLE: small trap hidden inside a much larger noisy graph.
        int k = 6;
        auto cnt = addTrap(0, k);
        int coreN = 2 * k;
        int extra = 24;
        int n = coreN + extra;
        // attach extra nodes with a random spanning path so the whole thing stays connected
        vector<int> order(extra);
        iota(order.begin(), order.end(), coreN);
        for (int i = extra - 1; i > 0; i--) swap(order[i], order[rnd.next(0, i)]);
        int prev = rnd.next(0, coreN - 1);
        for (int i = 0; i < extra; i++) { addEdge(prev, order[i], rnd.next(1, 2)); prev = order[i]; }
        addNoise(n, 1, 2, 40);
        int m = (int)edges.size();
        long long capNight = cnt.second + 20 + 4;
        long long capDay = m - capNight;
        emit(n, 2.6, capDay, capNight);
    } else if (testId == 7) {
        int k = 15;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 12);
        int m = (int)edges.size();
        long long capNight = cnt.second + 6 + 3;
        long long capDay = cnt.first + 6 + 3;
        emit(n, 2.7, capDay, capNight);
    } else if (testId == 8) {
        // PLANTED: two independent trap-pairs chained together with a light backbone.
        int k = 10;
        auto c1 = addTrap(0, k);
        auto c2 = addTrap(2 * k, k);
        int n = 4 * k;
        addEdge(k - 1, 2 * k, 1); // backbone connector between the two trap blocks (goes to day)
        addNoise(n, 1, 2, 10);
        int m = (int)edges.size();
        long long capNight = c1.second + c2.second + 5 + 3;
        long long capDay = m - capNight;
        emit(n, 2.9, capDay, capNight);
    } else if (testId == 9) {
        // TIGHT caps: the clean split (all of T1 to day) is INFEASIBLE -> forces adaptation.
        int k = 18;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 8);
        int m = (int)edges.size();
        long long capDay = cnt.first - 6;       // 6 short of holding all of T1
        long long capNight = m - capDay;
        emit(n, 3.0, capDay, capNight);
    } else {
        // largest / most adversarial: fills the constraint envelope.
        int k = 25;
        auto cnt = addTrap(0, k);
        int n = 2 * k;
        addNoise(n, 1, 2, 20);
        int m = (int)edges.size();
        long long capNight = cnt.second + 10 + 4;
        long long capDay = cnt.first + 10 + 4;
        emit(n, 3.1, capDay, capNight);
    }
    return 0;
}
