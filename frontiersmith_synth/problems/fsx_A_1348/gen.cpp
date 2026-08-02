// generator for "Splitting the Grove: k-Way Orchard Partition"
#include "testlib.h"
#include <vector>
#include <algorithm>
#include <set>
using namespace std;
typedef long long ll;
typedef pair<int,int> pii;

// random-attach recursive tree on `cnt` nodes with ids startId..startId+cnt-1.
vector<pii> randAttach(int startId, int cnt) {
    vector<pii> edges;
    for (int i = 1; i < cnt; i++) {
        int childId = startId + i;
        int parentOffset = rnd.next(0, i - 1);
        edges.push_back({startId + parentOffset, childId});
    }
    return edges;
}

// backbone path of length backboneLen (ids startId..startId+backboneLen-1), remaining
// cnt-backboneLen nodes attached as leaves uniformly at random on the backbone.
vector<pii> caterpillar(int startId, int cnt, int backboneLen) {
    vector<pii> edges;
    for (int i = 1; i < backboneLen; i++)
        edges.push_back({startId + i - 1, startId + i});
    for (int i = backboneLen; i < cnt; i++) {
        int attachOffset = rnd.next(0, backboneLen - 1);
        edges.push_back({startId + attachOffset, startId + i});
    }
    return edges;
}

// hub node at id `startId`, remaining cnt-1 nodes split into `groups` NEAR-EVEN random-
// attach branches (round-robin sizing, not a randomly concentrated split), each branch
// root directly joined to the hub. Hub ends with degree exactly `groups`.
vector<pii> starOfSubtrees(int startId, int cnt, int groups) {
    vector<pii> edges;
    int hub = startId;
    int remaining = cnt - 1;
    groups = min(groups, remaining);
    if (groups < 1) groups = 1;
    vector<int> sizes(groups, remaining / groups);
    for (int i = 0; i < remaining % groups; i++) sizes[i]++; // spread the remainder evenly
    int cur = startId + 1;
    for (int g = 0; g < groups; g++) {
        int sz = sizes[g];
        if (sz <= 0) continue;
        auto sub = randAttach(cur, sz);
        for (auto &e : sub) edges.push_back(e);
        edges.push_back({hub, cur});
        cur += sz;
    }
    return edges;
}

// Builds a hub block (hub + `groups` near-even branches of ~branchSize nodes each, total
// 1+groups*branchSize nodes, ids startId..startId+groups*branchSize) via starOfSubtrees,
// then wires the LAST node of the block (a leaf deep in the last branch, never the hub
// itself) to `connectTo` in the surrounding tree. This keeps the hub's degree EXACTLY
// `groups` regardless of the rest of the tree, and keeps every one of the hub's branches
// (plus the block-to-rest connection) close to `branchSize` -- so clearing the hub lands
// close to `groups` balanced pieces instead of one giant leftover plus tiny scraps.
int addHubBlock(vector<pii> &edges, int &nextId, int groups, int branchSize, int connectTo) {
    int startId = nextId;
    int total = 1 + groups * branchSize;
    auto block = starOfSubtrees(startId, total, groups);
    for (auto &e : block) edges.push_back(e);
    int lastNode = startId + total - 1;
    edges.push_back({lastNode, connectTo});
    nextId = startId + total;
    return startId; // hub id
}

int main(int argc, char **argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    int n, K, LAMBDA;
    vector<pii> edges;             // undirected edges (a,b), 1-indexed, size n-1
    set<int> hubs;                 // node ids to give a deliberately cheap clear cost
    int cLo = 5, cHi = 40;          // regular path cut cost range
    int pRegLo = 60, pRegHi = 150;  // regular (non-hub) clear cost range

    if (testId == 1) {
        n = 6; K = 2; LAMBDA = 2;
        edges = randAttach(1, n);
    } else if (testId == 2) {
        n = 16; K = 3; LAMBDA = 2;
        edges = randAttach(1, n);
    } else if (testId == 3) {
        // TRAP: one degree-6 hub, branches sized near the K-way target.
        K = 9; LAMBDA = 2;
        int base = 13;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h = addHubBlock(edges, nextId, 6, 5, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h);
    } else if (testId == 4) {
        // TRAP: two independent degree-5 hubs, both must be recognized (joint selection).
        K = 12; LAMBDA = 3;
        int base = 8;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h1 = addHubBlock(edges, nextId, 5, 4, rnd.next(1, base));
        int h2 = addHubBlock(edges, nextId, 5, 4, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h1); hubs.insert(h2);
    } else if (testId == 5) {
        // PLANTED: hub degree == K, branches near-even -> removing it lands almost exactly
        // on K balanced pieces in one bundled operation.
        n = 90; K = 5; LAMBDA = 2;
        edges = starOfSubtrees(1, n, K);
        hubs.insert(1);
    } else if (testId == 6) {
        // NEEDLE: one cheap degree-6 hub buried inside a larger, otherwise flat random tree.
        K = 8; LAMBDA = 2;
        int base = 30;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h = addHubBlock(edges, nextId, 6, 11, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h);
    } else if (testId == 7) {
        // Balance-sensitive TRAP, no hubs at all: pure joint-search-vs-sequential-centroid.
        n = 120; K = 10; LAMBDA = 3;
        edges = caterpillar(1, n, 45);
    } else if (testId == 8) {
        // Mixed: two degree-6 hubs plus a sizeable balance-sensitive random remainder.
        K = 14; LAMBDA = 3;
        int base = 16;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h1 = addHubBlock(edges, nextId, 6, 8, rnd.next(1, base));
        int h2 = addHubBlock(edges, nextId, 6, 8, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h1); hubs.insert(h2);
    } else if (testId == 9) {
        // Large bushy random tree with one moderately cheap degree-6 hub, plenty of noise,
        // and a large remaining local-search budget (M=7) to stress the joint pass at scale.
        K = 13; LAMBDA = 4;
        int base = 159;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h = addHubBlock(edges, nextId, 6, 15, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h);
    } else {
        // Largest adversarial: random backbone + three degree-5 hubs, envelope-filling
        // values (N and LAMBDA both at their stated maximum).
        K = 14; LAMBDA = 5;
        int base = 72;
        edges = randAttach(1, base);
        int nextId = base + 1;
        int h1 = addHubBlock(edges, nextId, 5, 15, rnd.next(1, base));
        int h2 = addHubBlock(edges, nextId, 5, 15, rnd.next(1, base));
        int h3 = addHubBlock(edges, nextId, 5, 15, rnd.next(1, base));
        n = nextId - 1;
        hubs.insert(h1); hubs.insert(h2); hubs.insert(h3);
        cLo = 8; cHi = 40; pRegLo = 100; pRegHi = 150;
    }

    if ((int)edges.size() != n - 1) {
        fprintf(stderr, "internal gen error: %d edges for n=%d\n", (int)edges.size(), n);
        return 1;
    }

    // keep total yield S roughly constant across test scales (so the quadratic balance
    // penalty stays commensurate with cutting cost at every n) by shrinking the per-plot
    // yield range as n grows.
    int wHi = max(1, min(6, (int)(700 / max(1, n))));
    if (wHi < 1) wHi = 1;

    vector<int> w(n + 1), p(n + 1);
    for (int v = 1; v <= n; v++) w[v] = rnd.next(1, wHi);

    vector<int> costs(edges.size());
    for (size_t i = 0; i < edges.size(); i++) costs[i] = rnd.next(cLo, cHi);
    // deliberately price every hub-incident path at the top of the range, so bundling them
    // into one clear is the only way to beat paying for them one at a time.
    int hubCLo = max(cLo, cHi - (cHi - cLo) / 4), hubCHi = cHi;
    for (size_t i = 0; i < edges.size(); i++) {
        if (hubs.count(edges[i].first) || hubs.count(edges[i].second))
            costs[i] = rnd.next(hubCLo, hubCHi);
    }

    vector<ll> sumEdgeCost(n + 1, 0);
    for (size_t i = 0; i < edges.size(); i++) {
        int a = edges[i].first, b = edges[i].second;
        sumEdgeCost[a] += costs[i];
        sumEdgeCost[b] += costs[i];
    }

    for (int v = 1; v <= n; v++) {
        if (hubs.count(v)) {
            double frac = rnd.next(0.10, 0.25);
            ll val = (ll)llround(sumEdgeCost[v] * frac);
            if (val < 1) val = 1;
            if (val > 150) val = 150;
            p[v] = (int)val;
        } else {
            p[v] = rnd.next(pRegLo, pRegHi);
        }
    }

    printf("%d %d %d\n", n, K, LAMBDA);
    for (int v = 1; v <= n; v++) printf("%d%c", w[v], v == n ? '\n' : ' ');
    for (int v = 1; v <= n; v++) printf("%d%c", p[v], v == n ? '\n' : ' ');
    for (size_t i = 0; i < edges.size(); i++)
        printf("%d %d %d\n", edges[i].first, edges[i].second, costs[i]);

    return 0;
}
