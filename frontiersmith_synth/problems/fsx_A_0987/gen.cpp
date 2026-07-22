#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

// ----------------------------------------------------------------------------
// Generator for "Quarrelsome Metronomes on a Coupled Table".
// testId is a difficulty/structure ladder: small sanity graphs -> bottleneck
// traps (barbells / chains of cliques / paths / grids) -> large adversarial
// bottleneck instances, interleaved with a few "control" (expander-ish /
// no-obvious-cut) graphs so greedy is not uniformly bad everywhere.
// ----------------------------------------------------------------------------

static void addEdge(set<pii> &E, int u, int v) {
    if (u == v) return;
    if (u > v) swap(u, v);
    E.insert({u, v});
}

static void cliqueEdges(set<pii> &E, int base, int size) {
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++)
            addEdge(E, base + i + 1, base + j + 1);
}

// random recursive spanning tree on nodes [base+1 .. base+size]
static void spanningTree(set<pii> &E, int base, int size) {
    for (int i = 2; i <= size; i++) {
        int parent = rnd.next(1, i - 1);
        addEdge(E, base + parent, base + i);
    }
}

// Count only edges INSIDE [base+1, base+size] -- using the global E.size() here
// would under-fill every cluster after the first one sharing the same set.
static int clusterEdgeCount(const set<pii> &E, int base, int size) {
    int c = 0;
    for (auto &e : E)
        if (e.first > base && e.first <= base + size && e.second > base && e.second <= base + size)
            c++;
    return c;
}

static void addRandomExtra(set<pii> &E, int base, int size, int targetTotal) {
    ll cap = (ll)size * (size - 1) / 2;
    if (targetTotal > cap) targetTotal = (int)cap;
    int attempts = 0, maxAttempts = targetTotal * 25 + 200;
    while (clusterEdgeCount(E, base, size) < targetTotal && attempts < maxAttempts) {
        attempts++;
        int u = base + rnd.next(1, size);
        int v = base + rnd.next(1, size);
        addEdge(E, u, v);
    }
}

// Top up any node under minDeg with extra random edges INSIDE [base+1,base+size],
// so uniform random sampling never leaves an accidental near-leaf inside a
// nominally "dense" cluster (which would plant an unintended micro-bottleneck).
static void ensureMinDegree(set<pii> &E, int base, int size, int minDeg) {
    if (size <= minDeg) return; // can't exceed a clique
    vector<int> deg(size + 1, 0);
    for (auto &e : E) {
        if (e.first > base && e.first <= base + size && e.second > base && e.second <= base + size) {
            deg[e.first - base]++;
            deg[e.second - base]++;
        }
    }
    for (int i = 1; i <= size; i++) {
        int guard = 0;
        while (deg[i] < minDeg && guard < minDeg * 20 + 50) {
            guard++;
            int j = rnd.next(1, size);
            if (j == i) continue;
            int u = base + i, v = base + j;
            if (u > v) swap(u, v);
            if (E.count({u, v})) continue;
            E.insert({u, v});
            deg[i]++; deg[j]++;
        }
    }
}

// sum-zero frequency multiset via +-v pairing, values in [1,maxV]
static vector<ll> genFreq(int n, int maxV) {
    vector<ll> f;
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        ll v = rnd.next(1, maxV);
        f.push_back(v);
        f.push_back(-v);
    }
    if (n % 2 == 1) f.push_back(0);
    for (int i = (int)f.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(f[i], f[j]);
    }
    return f;
}

// Relabel seats with a random permutation before printing, so raw seat-index
// order carries NO information about cluster/community structure (only the
// actual edges do). Without this, every construction step above numbers
// cluster 1 / cluster 2 / path position contiguously for convenience, and any
// solver that peeks at raw index (not real adjacency) could shortcut the
// intended graph-topology reasoning for free. The EDGE LIST ORDER is also
// shuffled (not just the endpoint labels): printing `E` in its internal
// build order would still list every intra-cluster edge contiguously before
// the bridge, letting a solver recover the communities from line position
// alone with no graph reasoning at all.
static void emit(int n, const set<pii> &E, const vector<ll> &f) {
    vector<int> perm(n + 1);
    for (int i = 1; i <= n; i++) perm[i] = i;
    for (int i = n; i >= 2; i--) {
        int j = rnd.next(1, i);
        swap(perm[i], perm[j]);
    }
    vector<pii> elist;
    elist.reserve(E.size());
    for (auto &e : E) elist.push_back({perm[e.first], perm[e.second]});
    for (int i = (int)elist.size() - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(elist[i], elist[j]);
    }
    printf("%d %d\n", n, (int)elist.size());
    for (auto &e : elist) printf("%d %d\n", e.first, e.second);
    for (int i = 0; i < n; i++) printf("%lld%c", f[i], i + 1 < n ? ' ' : '\n');
}

int main(int argc, char *argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n;
    set<pii> E;

    if (testId == 1) {
        // tiny sanity: 6-cycle
        n = 6;
        for (int i = 1; i <= n; i++) addEdge(E, i, i % n + 1);
        emit(n, E, genFreq(n, 20));

    } else if (testId == 2) {
        // barbell: two K7 joined by a single bridge edge -- TRAP
        n = 14;
        cliqueEdges(E, 0, 7);
        cliqueEdges(E, 7, 7);
        addEdge(E, 7, 8);
        emit(n, E, genFreq(n, 200));

    } else if (testId == 3) {
        // path graph: every edge is a cut -- TRAP (extreme)
        n = 24;
        for (int i = 1; i < n; i++) addEdge(E, i, i + 1);
        emit(n, E, genFreq(n, 500));

    } else if (testId == 4) {
        // two moderately-dense random clusters joined by ONE bridge edge -- TRAP
        n = 30;
        int c1 = 15, c2 = 15;
        spanningTree(E, 0, c1);
        addRandomExtra(E, 0, c1, 30);
        ensureMinDegree(E, 0, c1, 6);
        spanningTree(E, c1, c2);
        addRandomExtra(E, c1, c2, 30);
        ensureMinDegree(E, c1, c2, 6);
        addEdge(E, c1, c1 + 1);
        emit(n, E, genFreq(n, 300));

    } else if (testId == 5) {
        // cycle + random chords -- semi-control
        n = 40;
        for (int i = 1; i <= n; i++) addEdge(E, i, i % n + 1);
        addRandomExtra(E, 0, n, n + 15);
        emit(n, E, genFreq(n, 400));

    } else if (testId == 6) {
        // 7x7 grid -- spectral trap (smooth Fiedler mode along one axis)
        int rows = 7, cols = 7;
        n = rows * cols;
        auto id = [&](int r, int c) { return r * cols + c + 1; };
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                if (c + 1 < cols) addEdge(E, id(r, c), id(r, c + 1));
                if (r + 1 < rows) addEdge(E, id(r, c), id(r + 1, c));
            }
        emit(n, E, genFreq(n, 500));

    } else if (testId == 7) {
        // chain of 4 cliques joined by 3 single bridge edges -- multi-bottleneck TRAP
        n = 60;
        int k = 4, sz = 15;
        for (int b = 0; b < k; b++) cliqueEdges(E, b * sz, sz);
        for (int b = 0; b + 1 < k; b++) addEdge(E, b * sz + 1, (b + 1) * sz + 1);
        emit(n, E, genFreq(n, 600));

    } else if (testId == 8) {
        // fairly dense random connected graph -- control (expander-ish, no sharp cut)
        n = 80;
        spanningTree(E, 0, n);
        addRandomExtra(E, 0, n, 800);
        ensureMinDegree(E, 0, n, 8);
        emit(n, E, genFreq(n, 700));

    } else if (testId == 9) {
        // large path -- TRAP, fills envelope on size
        n = 120;
        for (int i = 1; i < n; i++) addEdge(E, i, i + 1);
        emit(n, E, genFreq(n, 900));

    } else { // testId == 10
        // largest adversarial instance: two big dense clusters, ONE bridge edge,
        // widest frequency range -- TRAP / needle
        n = 150;
        int c1 = 75, c2 = 75;
        spanningTree(E, 0, c1);
        addRandomExtra(E, 0, c1, 1450);
        ensureMinDegree(E, 0, c1, 12);
        spanningTree(E, c1, c2);
        addRandomExtra(E, c1, c2, 1450);
        ensureMinDegree(E, c1, c2, 12);
        addEdge(E, c1, c1 + 1);
        emit(n, E, genFreq(n, 1000));
    }

    return 0;
}
