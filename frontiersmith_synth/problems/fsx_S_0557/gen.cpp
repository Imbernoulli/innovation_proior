#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// ---------------------------------------------------------------------------
// Ignite the Idea: clustered seeding in a threshold (bootstrap-percolation)
// network.  We plant:
//   * CLIQUE clusters (the nucleation targets): with r=2, seeding an adjacent
//     PAIR inside a clique ignites the whole clique (a seeded 2-core cascades).
//   * a CHAIN of gateway bridges between consecutive clusters: once cluster i
//     is fully active, a designated gateway node of cluster i+1 gets 2 active
//     neighbours and auto-activates -- so the *optimum* can ignite a chained
//     cluster with only ONE extra seed (synergy / ordering), leaving headroom
//     above any "2-seeds-per-cluster" heuristic.
//   * ANCHOR clusters: two clique nodes given the globally-highest degree, so
//     the obvious "seed the highest-degree people" heuristic wastes its budget
//     igniting only these few, then dumps the rest on hubs.
//   * HUB stars (the trap / noise): a centre wired to many degree-1 leaves.
//     Seeding a hub never spreads (each leaf sees only one adopter < r), yet a
//     hub has higher degree than an ordinary clique member.
// Node labels are permuted at the end so only DEGREE (not index) carries signal.
// ---------------------------------------------------------------------------

int cur = 0;
int newNode() { return ++cur; }

vector<pair<int,int>> edges;
void addEdge(int a, int b) { edges.push_back({a, b}); }

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    const int r = 2;

    int numClusters, szmin, szmax, nAnchor;
    if (testId == 1) {                       // tiny -- matches the example scale
        numClusters = 5;  szmin = 4;  szmax = 7;  nAnchor = 1;
    } else {
        numClusters = 5 + 4 * (testId - 1);  // 9,13,...,41
        szmin = 8;  szmax = 16;  nAnchor = 2;
        // planted-structure variety across the ladder:
        if (testId % 3 == 0) { szmin = 10; szmax = 18; }   // fatter cores
        if (testId % 4 == 0) { szmin = 6;  szmax = 20; }   // high size skew
    }

    // cluster sizes
    vector<int> sz(numClusters);
    int maxSize = 0;
    for (int j = 0; j < numClusters; j++) {
        sz[j] = rnd.next(szmin, szmax);
        maxSize = max(maxSize, sz[j]);
    }

    int k         = numClusters;             // seed budget
    int hubLeaves = maxSize + 3;             // hub degree > any ordinary cluster node
    int numHubs   = k + 2;                   // enough hubs to absorb the degree-greedy budget
    int anchorDeg = hubLeaves + 4;           // strictly the highest degree in the graph

    // ---- build clusters (cliques) ----
    vector<vector<int>> C(numClusters);
    for (int j = 0; j < numClusters; j++) {
        for (int i = 0; i < sz[j]; i++) C[j].push_back(newNode());
        for (int a = 0; a < sz[j]; a++)
            for (int b = a + 1; b < sz[j]; b++)
                addEdge(C[j][a], C[j][b]);
    }

    // ---- synergy bridges (DISJOINT pairs, deliberately NON-transitive) ----
    // Pair up clusters (0,1),(2,3),... In pair (a,b) the "gateway" node0 of the
    // odd cluster b is wired to node0,node1 of the even cluster a. node0/node1 of
    // a are adjacent (clique) so the gateway sits in a triangle and auto-activates
    // once a is fully active -- i.e. after igniting a, cluster b can be ignited
    // with a SINGLE extra seed instead of two. Because odd clusters never gateway
    // anything, this saving does NOT chain: a clever allocator gains ~1 seed per
    // pair (real headroom above a 2-seeds-per-cluster heuristic), but no cascade
    // ever runs away across the whole graph.
    for (int j = 1; j < numClusters; j += 2) {
        int g = C[j][0];
        addEdge(g, C[j - 1][0]);
        addEdge(g, C[j - 1][1]);
    }

    // ---- anchor clusters: raise two NON-gateway nodes (index 2,3) to the top degree ----
    vector<int> anchorPos;
    if (nAnchor == 1) {
        anchorPos = { numClusters / 2 };
    } else {
        anchorPos = { 1, numClusters - 2 };  // spaced, non-adjacent in the chain
    }
    for (int p : anchorPos) {
        for (int t = 2; t <= 3; t++) {       // clusters always have size >= 4
            int node = C[p][t];
            int have = 0;
            for (auto &e : edges) if (e.first == node || e.second == node) have++;
            int need = anchorDeg - have;
            for (int x = 0; x < need; x++) {
                int leaf = newNode();
                addEdge(node, leaf);
            }
        }
    }

    // ---- hub stars (traps): centre + many degree-1 leaves ----
    for (int h = 0; h < numHubs; h++) {
        int centre = newNode();
        for (int l = 0; l < hubLeaves; l++) {
            int leaf = newNode();
            addEdge(centre, leaf);
        }
    }

    int n = cur;
    int m = (int)edges.size();

    // ---- permute labels so index order leaks nothing; shuffle edge order ----
    vector<int> lab(n + 1);
    for (int i = 1; i <= n; i++) lab[i] = i;
    for (int i = n; i >= 2; i--) {
        int j = rnd.next(1, i);
        swap(lab[i], lab[j]);
    }
    shuffle(edges.begin(), edges.end());

    printf("%d %d %d %d\n", n, m, r, k);
    for (auto &e : edges) {
        int a = lab[e.first], b = lab[e.second];
        printf("%d %d\n", a, b);
    }
    return 0;
}
