#include "testlib.h"
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // difficulty / structure ladder
    static const int nArr[11]  = {0, 30, 60, 150, 300, 600, 1000, 1500, 2500, 4000, 5000};
    static const int degArr[11]= {0, 6,   6,  7,   7,   8,    8,    7,    8,    8,    9};
    int n = nArr[testId];
    int avgDeg = degArr[testId];

    // trap tests: engineered decoy hubs that fool a degree/edge-local greedy
    bool trapTest = (testId == 4 || testId == 6 || testId == 8 || testId == 9 || testId == 10);
    // imbalance tests: true communities are not exactly half/half
    bool imbalanceTest = (testId == 6 || testId == 10);
    // needle/control test: very clean, low-noise separation, no hubs
    bool cleanTest = (testId == 7);

    int c0size;
    if (imbalanceTest) c0size = max(3, (int)(n * 0.42));
    else c0size = n / 2;
    if (c0size < 2) c0size = 2;
    if (c0size > n - 2) c0size = n - 2;
    int c1size = n - c0size;

    // random permutation assigns vertex IDs to communities (NOT sorted by id)
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i + 1;
    for (int i = n - 1; i > 0; i--) {
        int j = rnd.next(0, i);
        swap(perm[i], perm[j]);
    }
    vector<int> members0, members1;
    members0.reserve(c0size);
    members1.reserve(c1size);
    vector<int> commOf(n + 1, 0);
    for (int i = 0; i < n; i++) {
        int v = perm[i];
        if (i < c0size) { members0.push_back(v); commOf[v] = 0; }
        else { members1.push_back(v); commOf[v] = 1; }
    }

    set<pair<int,int>> edgeset;
    vector<int> Eu, Ev, Ew;
    auto addEdge = [&](int u, int v, int w) {
        if (u == v) return;
        if (u > v) swap(u, v);
        pair<int,int> key(u, v);
        if (edgeset.count(key)) return;
        edgeset.insert(key);
        Eu.push_back(u); Ev.push_back(v); Ew.push_back(w);
    };

    // 1) spanning tree per community: guarantees connectivity + min degree >= 1
    auto spanningTree = [&](vector<int>& mem) {
        for (size_t i = 1; i < mem.size(); i++) {
            int j = rnd.next(0, (int)i - 1);
            int w = rnd.next(1, 3);
            addEdge(mem[i], mem[j], w);
        }
    };
    spanningTree(members0);
    spanningTree(members1);

    // 2) extra within-community density (weak/normal edges)
    auto extraWithin = [&](vector<int>& mem, int deg) {
        int sz = (int)mem.size();
        if (sz < 2) return;
        long long attempts = (long long)sz * deg / 2;
        for (long long t = 0; t < attempts; t++) {
            int a = mem[rnd.next(0, sz - 1)];
            int b = mem[rnd.next(0, sz - 1)];
            int w = rnd.next(1, 3);
            addEdge(a, b, w);
        }
    };
    extraWithin(members0, avgDeg);
    extraWithin(members1, avgDeg);

    // 3) background cross-community noise (weak true bridges, the honest signal)
    {
        double noiseFrac = cleanTest ? 0.25 : 0.60;
        long long noiseEdges = (long long)(n * noiseFrac);
        for (long long t = 0; t < noiseEdges; t++) {
            int a = members0[rnd.next(0, (int)members0.size() - 1)];
            int b = members1[rnd.next(0, (int)members1.size() - 1)];
            addEdge(a, b, rnd.next(1, 2));
        }
    }

    // 4) decoy hubs: a handful of vertices, ALL drawn from ONE true community,
    //    each wired to a heavy random slice of the OTHER community. Putting every
    //    hub on the same side (rather than splitting hubs across both sides) is
    //    what actually traps a "seed from the two highest-degree, mutually
    //    non-adjacent vertices" greedy: since no hub-hub edges are ever added,
    //    two same-community hubs are non-adjacent by construction, so that
    //    greedy's seed-selection rule anchors BOTH of its growth seeds in the
    //    SAME true community -- corrupting the side0/side1 premise from the
    //    very first step, before any flood-fill even happens. Splitting hubs
    //    symmetrically across both communities (the earlier design) mainly adds
    //    unavoidable structural cost that a correct answer must also pay, and
    //    empirically failed to hurt greedy specifically.
    if (trapTest) {
        int hubCommunitySide = (testId % 2 == 0) ? 0 : 1;
        vector<int>& hubPool = (hubCommunitySide == 0) ? members0 : members1;
        vector<int>& targetPool = (hubCommunitySide == 0) ? members1 : members0;

        int hubCount = max(3, n / 150);
        vector<int> pool = hubPool;
        for (int i = (int)pool.size() - 1; i > 0; i--) { int j = rnd.next(0, i); swap(pool[i], pool[j]); }
        hubCount = min(hubCount, (int)pool.size() / 2);
        hubCount = max(hubCount, 2);

        double touchFrac = 0.12;
        for (int h = 0; h < hubCount; h++) {
            int hub = pool[h];
            int targets = max(4, (int)(targetPool.size() * touchFrac));
            vector<int> tgt = targetPool;
            for (int i = (int)tgt.size() - 1; i > 0 && i >= (int)tgt.size() - targets; i--) {
                int j = rnd.next(0, i);
                swap(tgt[i], tgt[j]);
            }
            int lim = min((int)tgt.size(), targets);
            for (int i = 0; i < lim; i++) {
                addEdge(hub, tgt[(int)tgt.size() - 1 - i], rnd.next(6, 10));
            }
        }
    }

    int m = (int)Eu.size();
    int lo = (int)(n * 0.35);
    if (lo < 1) lo = 1;
    int hi = n - lo;
    if (hi >= n) hi = n - 1;
    if (lo > hi) lo = hi;

    printf("%d %d %d %d\n", n, m, lo, hi);
    for (int i = 0; i < m; i++) {
        printf("%d %d %d\n", Eu[i], Ev[i], Ew[i]);
    }
    return 0;
}
