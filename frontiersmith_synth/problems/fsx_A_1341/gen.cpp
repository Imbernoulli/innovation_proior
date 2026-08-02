#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "The Tangled Arcade" (grundy-value-compute / sum-of-games / move-graph-cycles).
//
// Each test file bundles M independent POSITION INSTANCES. Each instance is a disjunctive
// sum of C "live" components (each with a token / active node) plus D "decoy" components
// (no token, pure noise). Every component is a small directed graph built on its OWN local
// node range; components never share nodes or edges (true vertex-disjoint sum), but the
// whole instance is printed as ONE big node/edge list with node ids globally permuted and
// edge order globally shuffled -- the interleaving hides which nodes belong to which
// component. A solver who does not recognize the decomposition and instead tries to search
// the joint state space (product of component sizes -> exponential in C) will blow up on the
// larger tests; a solver who realizes each component's contribution can be computed locally
// (independent of the others) and combined by XOR only needs O(V+E) work.
//
// Each live component is built in one of two modes, chosen per-component:
//   * CYCLIC:  nodes 0..L-1 form a directed ring (L in [1,4]; L=1 is a self-loop) and the
//              token sits ON the ring -> by this problem's rule such a component is JAMMED
//              (contributes 0). Extra tail nodes hang off by forward-only edges.
//   * ACYCLIC: the whole component is a genuine DAG (edges only go from a lower to a higher
//              local index in a fixed random topological order) -- ordinary Sprague-Grundy
//              mex applies, token can sit anywhere.
// A small planted fraction of instances are single isolated tokens (0 out-degree) so the
// "always PASS" baseline has a guaranteed-positive, controlled hit rate on every test.

struct Params {
    int M;                 // instances in this test
    int Cmin, Cmax;         // live components per instance
    int Dmin, Dmax;         // decoy components per instance
    int Smin, Smax;         // component size (nodes)
    int extraMax;           // extra random forward edges per component
    int pCycPct;            // % chance a live component is CYCLIC
    int pStuckPct;          // % chance an instance is a planted isolated-token instance
};

Params setparams(int t) {
    Params p;
    switch (t) {
        case 1:  p = {150, 1, 2, 0, 0, 3, 5,  1, 25, 16}; break;  // tiny warm-up
        case 2:  p = {200, 2, 2, 0, 1, 4, 6,  2, 30, 16}; break;
        case 3:  p = {220, 2, 3, 0, 1, 4, 6,  2, 32, 16}; break;
        case 4:  p = {240, 3, 3, 0, 1, 5, 7,  2, 35, 16}; break;
        case 5:  p = {260, 3, 4, 0, 2, 5, 8,  3, 40, 15}; break;  // near budget edge
        case 6:  p = {280, 4, 5, 0, 2, 6, 8,  3, 42, 15}; break;  // trap begins
        case 7:  p = {300, 4, 6, 1, 3, 6, 9,  3, 45, 15}; break;  // trap
        case 8:  p = {320, 5, 7, 1, 3, 7, 9,  3, 48, 15}; break;  // trap
        case 9:  p = {340, 6, 8, 2, 4, 7, 10, 4, 50, 15}; break;  // trap
        default: p = {360, 7, 9, 2, 4, 8, 10, 4, 50, 15}; break;  // trap, full envelope
    }
    return p;
}

struct Component {
    int size;
    vector<pair<int,int>> edges; // local (u,v)
    int tok;                     // -1 if decoy (no active token)
};

// Build one component. If forceCyclic, nodes [0,L) form a directed ring containing the
// token; remaining nodes attach by forward-only edges. Otherwise the whole component is a
// forward-only DAG (edges (i,j) only with i<j in local index order).
Component genComponent(int gs, bool forceCyclic, bool wantToken) {
    Component c;
    c.size = gs;
    c.tok = -1;
    int L = 0;
    if (forceCyclic) {
        L = rnd.next(1, min(gs, 4));
        for (int i = 0; i < L; i++) {
            int j = (i + 1) % L;
            c.edges.push_back({i, j});
        }
        if (wantToken) c.tok = rnd.next(0, L - 1);
    } else {
        if (wantToken) c.tok = rnd.next(0, gs - 1);
    }
    // forward-only attachment for the remaining nodes (guarantees weak connectivity;
    // guarantees NO new cycle is created outside the ring since edges only go low->high
    // index except the ring's own closing edge above).
    for (int i = max(L, 1); i < gs; i++) {
        int par = rnd.next(0, i - 1);
        c.edges.push_back({par, i});
    }
    // extra forward branching edges among indices >= L (adds richer out-degree / mex
    // variety without ever creating an unintended cycle).
    int extra = rnd.next(0, min(3, max(0, gs - 2)));
    for (int k = 0; k < extra; k++) {
        if (gs - L < 2) break;
        int a = rnd.next(L, gs - 2);
        int b = rnd.next(a + 1, gs - 1);
        c.edges.push_back({a, b});
    }
    if (!wantToken) c.tok = -1;
    return c;
}

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);
    Params P = setparams(t);

    printf("%d\n", P.M);
    for (int m = 0; m < P.M; m++) {
        bool stuck = rnd.next(1, 100) <= P.pStuckPct;
        vector<Component> comps;
        if (stuck) {
            Component c;
            c.size = 1;
            c.tok = 0; // no edges at all -> zero out-degree -> guaranteed no legal move
            comps.push_back(c);
        } else {
            int C = rnd.next(P.Cmin, P.Cmax);
            int D = (P.Dmax > 0) ? rnd.next(P.Dmin, P.Dmax) : 0;
            for (int i = 0; i < C; i++) {
                int gs = rnd.next(P.Smin, P.Smax);
                bool cyc = rnd.next(1, 100) <= P.pCycPct;
                comps.push_back(genComponent(gs, cyc, true));
            }
            for (int i = 0; i < D; i++) {
                int gs = rnd.next(max(2, P.Smin - 1), P.Smax);
                bool cyc = rnd.next(1, 100) <= P.pCycPct;
                comps.push_back(genComponent(gs, cyc, false));
            }
        }

        // assign global ids: build local->global permutation across the whole instance
        int n = 0;
        for (auto& c : comps) n += c.size;
        vector<int> perm(n);
        for (int i = 0; i < n; i++) perm[i] = i;
        for (int i = n - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);

        vector<pair<int,int>> allEdges;
        vector<int> tokens;
        int base = 0;
        for (auto& c : comps) {
            for (auto& e : c.edges) allEdges.push_back({perm[base + e.first], perm[base + e.second]});
            if (c.tok >= 0) tokens.push_back(perm[base + c.tok]);
            base += c.size;
        }
        // shuffle edge print order and token print order (further hides component identity)
        for (int i = (int)allEdges.size() - 1; i > 0; i--) swap(allEdges[i], allEdges[rnd.next(0, i)]);
        for (int i = (int)tokens.size() - 1; i > 0; i--) swap(tokens[i], tokens[rnd.next(0, i)]);

        int c_count = (int)tokens.size();
        int e_count = (int)allEdges.size();
        printf("%d %d %d\n", n, c_count, e_count);
        for (int i = 0; i < c_count; i++) printf("%d%c", tokens[i], i + 1 == c_count ? '\n' : ' ');
        for (auto& e : allEdges) printf("%d %d\n", e.first, e.second);
    }
    return 0;
}
