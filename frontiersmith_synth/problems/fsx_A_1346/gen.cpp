#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// gen.cpp -- "Corner Patrol: Dismantling a Graph to Trap an Evader"
//
// Builds a connected graph as: a random backbone TREE rooted at vertex 0, whose first
// P vertices (1..P) are forced to be DIRECT CHILDREN of the root (a "shallow pool",
// mutually non-adjacent by construction, all at BFS-depth 1), plus a modest number of
// random extra backbone edges (never connecting two shallow-pool vertices to each
// other, so the pool stays an independent set), plus PLANTED "corner gadgets": pick a
// small non-adjacent set of hubs from the shallow pool, connect each hub to several
// OTHER shallow-pool vertices (its private "decoys", via extra edges), then attach many
// brand-new leaf vertices to ALL hubs of the gadget AND to all its decoys. A leaf has
// degree (hubs + decoys), yet is provably dominated by the hubs ALONE (decoys are
// already covered by a hub's own neighborhood) -- so the true minimum guard is just the
// hub set, while "guard with every live neighbor" (the naive baseline / the textbook
// single-dominator corner test, which requires exactly ONE neighbor to cover
// everything and always fails here since >=2 non-adjacent hubs are both required) pays
// for every decoy too. Because all gadget vertices sit at BFS-depth 2 and every hub /
// decoy sits at depth <=1, they are still fully live (unprocessed) under the deepest-
// first elimination order used throughout this problem family -- the trap is guaranteed
// to engage regardless of solver strategy shape.
//
// testId is a difficulty/structure ladder: 1 is a tiny, gadget-free sanity case; 2..10
// grow the backbone and progressively add more / bigger gadgets (2-hub and 3-hub mixed)
// so >=3 of the 10 tests are genuine traps, filling the stated size envelope by test 10.

struct EdgeSet {
    set<pair<int,int>> es;
    bool has(int a,int b) const { if(a>b) swap(a,b); return es.count({a,b}) != 0; }
    bool add(int a,int b){ if(a==b) return false; if(a>b) swap(a,b); return es.insert({a,b}).second; }
};

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);
    int idx = min(max(testId, 1), 10);

    // ---- backbone size + shallow-pool size per testId ----
    // P must comfortably exceed the TOTAL (hubs+decoys) needed by all gadgets planted in
    // this test (see below) since every gadget draws a DISJOINT slice of the pool -- two
    // gadgets sharing a pool vertex would silently wire an edge between what was meant to
    // be a non-adjacent hub pair (via that vertex's other gadget's hub-decoy edges).
    int backboneN[10] = {18, 26, 34, 50, 64, 84, 110, 140, 170, 210};
    int poolP[10]      = {6,   8,  9, 16, 17, 18,  24,  26,  28,  30};
    int nBackbone = backboneN[idx-1];
    int P = poolP[idx-1];

    EdgeSet E;
    EdgeSet treeE;
    vector<int> parent(nBackbone, -1);
    vector<int> depth(nBackbone, 0);

    // vertices 1..P: forced direct children of root 0 (shallow pool, mutually non-adjacent)
    for (int i = 1; i <= P; i++) {
        parent[i] = 0; depth[i] = 1;
        E.add(0, i); treeE.add(0, i);
    }
    // remaining backbone vertices: random tree attachment (parent < i), realistic branching
    for (int i = P + 1; i < nBackbone; i++) {
        int p = rnd.next(0, i - 1);
        parent[i] = p; depth[i] = depth[p] + 1;
        E.add(p, i); treeE.add(p, i);
    }

    // extra random backbone edges (never both endpoints inside the shallow pool, so the
    // pool stays available as a source of mutually-non-adjacent hubs / decoys)
    int extra = nBackbone / 7;
    int guard = 0;
    while (extra > 0 && guard < 4000) {
        guard++;
        int a = rnd.next(0, nBackbone - 1);
        int b = rnd.next(0, nBackbone - 1);
        if (a == b) continue;
        if (a <= P && b <= P) continue; // keep shallow pool independent
        if (E.add(a, b)) extra--;
    }

    // "easy corner" edges: for most non-pool backbone vertices with depth>=2, also
    // connect them to their own grandparent (already an edge of their parent). This
    // gives them alive-degree 2 (parent + grandparent) under deepest-first elimination,
    // yet a SINGLE dominator (the parent, whose own neighborhood already contains the
    // grandparent) always exists -- so a real single-guard corner-check (greedy) finds
    // a cheap size-1 certificate here while "guard with every live neighbor" (trivial)
    // wastefully pays for 2. This is what gives greedy a genuine, broad edge over
    // trivial outside of the (much harder) planted gadgets.
    for (int i = P + 1; i < nBackbone; i++) {
        if (depth[i] < 2) continue;
        if (rnd.next(100) >= 96) continue; // ~96% density, keep a few plain degree-1 vertices too
        int gp = parent[parent[i]];
        E.add(i, gp);
        // occasionally also reach one hop further (great-grandparent, already adjacent to
        // the grandparent) -- still a single dominator (the parent's neighborhood covers
        // it transitively is NOT required; the parent must cover it directly, so only add
        // this when the parent is itself adjacent to the great-grandparent, i.e. depth>=3
        // and parent==grandparent's child through the tree, which is automatic) to widen
        // the trivial-vs-greedy gap further on deeper vertices.
        if (depth[i] >= 3 && rnd.next(100) < 50) {
            int ggp = parent[gp];
            if (E.has(parent[i], ggp)) E.add(i, ggp);
        }
    }

    // ---- planted corner gadgets (trap cases): testId 2..10 ----
    // each gadget: pick H hubs (2 or 3) + D decoys, all distinct, from the shallow pool
    // [1..P]; connect each hub to each decoy (extra edges); then add L brand-new leaves,
    // each adjacent to every hub and every decoy of this gadget.
    long long n = nBackbone;

    struct Gadget { int H, D, L; };
    vector<Gadget> gadgets;
    if (idx == 1) {
        // pure sanity case: no gadgets
    } else if (idx == 2) {
        gadgets = {{2, 3, 10}};
    } else if (idx == 3) {
        gadgets = {{3, 3, 12}};
    } else if (idx <= 6) {
        gadgets = {{2, 4, 10 + 3*idx}, {3, 3, 8 + 2*idx}};
    } else {
        gadgets = {{2, 5, 10 + 2*idx}, {3, 4, 8 + 2*idx}, {2, 3, 6 + 2*idx}};
    }

    vector<vector<int>> leafAdj; // adjacency (hub/decoy ids) for each new leaf vertex, in creation order

    // shuffle the WHOLE pool once, then hand each gadget a DISJOINT slice of it (cursor
    // advances by H+D per gadget) so no pool vertex ever plays a role in two gadgets.
    vector<int> sharedPool;
    for (int i = 1; i <= P; i++) sharedPool.push_back(i);
    shuffle(sharedPool.begin(), sharedPool.end());
    int poolCursor = 0;

    for (auto &g : gadgets) {
        int need = g.H + g.D;
        if (poolCursor + need > (int)sharedPool.size()) continue; // not enough disjoint pool left, skip
        vector<int> hubs(sharedPool.begin() + poolCursor, sharedPool.begin() + poolCursor + g.H);
        vector<int> decoys(sharedPool.begin() + poolCursor + g.H, sharedPool.begin() + poolCursor + need);
        poolCursor += need;
        if ((int)hubs.size() < 2 || decoys.empty()) continue; // degenerate, skip gadget

        // connect every hub to every decoy (extra edges; dedupe against existing)
        for (int h : hubs)
            for (int d : decoys)
                E.add(h, d);

        // add L new leaf vertices, each adjacent to every hub and every decoy
        for (int i = 0; i < g.L; i++) {
            vector<int> adj;
            for (int h : hubs) adj.push_back(h);
            for (int d : decoys) adj.push_back(d);
            leafAdj.push_back(adj);
        }
    }

    n = nBackbone + (long long)leafAdj.size();

    // ---- assemble final edge list: tree edges + non-tree E edges + leaf edges ----
    vector<pair<int,int>> edges;
    for (int i = 1; i < nBackbone; i++) edges.push_back({parent[i], i});
    for (auto &pr : E.es) if (!treeE.has(pr.first, pr.second)) edges.push_back(pr);
    int leafStart = nBackbone;
    for (int li = 0; li < (int)leafAdj.size(); li++) {
        int leafId = leafStart + li;
        for (int nb : leafAdj[li]) edges.push_back({leafId, nb});
    }

    long long m = (long long)edges.size();
    printf("%lld %lld\n", n, m);
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);

    return 0;
}
