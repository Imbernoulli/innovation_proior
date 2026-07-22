#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// "Charging Depots for the Bridge City"   family: commute-time-depots
//
// City = several DENSE BLOCKS (moderately-dense random subgraphs, decent hop-
// diameter but strong internal redundancy / low resistance) joined by THIN
// BRIDGES (simple chains of degree-2 nodes, no redundancy at all -- resistance
// there grows exactly as fast as hop distance, but the true random-walk ACCESS
// TIME along a corridor with no depot at either end grows QUADRATICALLY in its
// length, not linearly). testId ladder scales size and, for the "trap" tests,
// deliberately stretches thin bridges / dead-end pendant corridors far beyond
// what a hop-distance facility-location instinct would think to fear.
//
// Vertices are 1-indexed. Output format:
//   n m k
//   p_1 p_2 ... p_n            (population weight of each hub, 1..25)
//   u_1 v_1
//   ...
//   u_m v_m
// -----------------------------------------------------------------------------

static int nextId;
static vector<pair<int,int>> edges;
static vector<int> pop_; // pop_[v] valid for v in [1, nextId-1]; grown via ensure()

static void ensurePop(int v){
    if ((int)pop_.size() <= v) pop_.resize(v + 1, 0);
}
static void setPop(int v, int p){ ensurePop(v); pop_[v] = p; }

// Build a connected, moderately-dense random blob of `size` NEW vertices
// (fresh ids starting at nextId). Guarantees connectivity via a random
// spanning path, then adds extra edges to reach an average degree of
// roughly `avgDeg` (kept moderate ~3.5-5.5 on purpose: real hop-diameter,
// but strong path-redundancy -> low effective resistance/access time).
// Returns the ids assigned, in order.
vector<int> buildBlob(int size, double avgDeg){
    vector<int> ids(size);
    for (int i = 0; i < size; i++){ ids[i] = nextId++; setPop(ids[i], rnd.next(5, 25)); }
    if (size == 1) return ids;

    vector<int> perm(size);
    for (int i = 0; i < size; i++) perm[i] = i;
    for (int i = size - 1; i > 0; i--) swap(perm[i], perm[rnd.next(0, i)]);
    set<pair<int,int>> used;
    auto addLocal = [&](int a, int b){
        if (a == b) return;
        int lo = min(a, b), hi = max(a, b);
        if (used.count({lo, hi})) return;
        used.insert({lo, hi});
        edges.push_back({ids[lo], ids[hi]});
    };
    for (int i = 0; i + 1 < size; i++) addLocal(perm[i], perm[i + 1]);

    long long targetEdges = (long long)llround(avgDeg * size / 2.0);
    targetEdges = max<long long>(targetEdges, size - 1);
    long long maxPossible = (long long)size * (size - 1) / 2;
    targetEdges = min(targetEdges, maxPossible);

    vector<pair<int,int>> allPairs;
    allPairs.reserve(maxPossible);
    for (int i = 0; i < size; i++)
        for (int j = i + 1; j < size; j++) allPairs.push_back({i, j});
    for (int i = (int)allPairs.size() - 1; i > 0; i--) swap(allPairs[i], allPairs[rnd.next(0, i)]);
    for (auto &pr : allPairs){
        if ((long long)used.size() >= targetEdges) break;
        addLocal(pr.first, pr.second);
    }
    return ids;
}

// Build a dead-end pendant chain (no far endpoint) of `length` vertices
// starting from existing vertex `fromId`.
void buildPendant(int fromId, int length){
    int prev = fromId;
    for (int i = 0; i < length; i++){
        int v = nextId++;
        setPop(v, rnd.next(1, 3));
        edges.push_back({prev, v});
        prev = v;
    }
}

void emit(int n, int k){
    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, k);
    for (int v = 1; v <= n; v++) printf("%d%c", pop_[v], v == n ? '\n' : ' ');
    for (auto &e : edges) printf("%d %d\n", e.first, e.second);
}

int main(int argc, char* argv[]){
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    nextId = 1;
    edges.clear();
    pop_.assign(1, 0);

    if (testId == 1){
        // Hand-checkable tiny example: blob A (4, near-clique) -- bridge (3
        // interior chain nodes) -- blob B (3, triangle+extra).
        pop_.assign(11, 0);
        int a[4] = {1,2,3,4};
        for (int i = 0; i < 4; i++) pop_[a[i]] = 10;
        edges = {{1,2},{2,3},{3,4},{1,3},{2,4}};
        pop_[5] = 2; pop_[6] = 2; pop_[7] = 2;
        edges.push_back({4,5}); edges.push_back({5,6}); edges.push_back({6,7});
        int b[3] = {8,9,10};
        for (int i = 0; i < 3; i++) pop_[b[i]] = 10;
        edges.push_back({7,8});
        edges.push_back({8,9}); edges.push_back({9,10}); edges.push_back({8,10});
        nextId = 11;
        emit(10, 1);
        return 0;
    }

    // Parameter table per testId: Bc blobs joined in a PATH by bridgeLens
    // (Bc-1 entries), and each blob may additionally sprout dead-end
    // pendant arms of its own (armLens[bi]). Bc==1 with no bridges and
    // several arms degenerates to a pure "star" of pendant corridors; mixing
    // path-bridges and pendant arms on the SAME blobs forces a combined
    // decision: how many depots to spend covering each district at all
    // (blob-collapse tradeoff) vs. how far to reach along which corridor
    // (tendril-length tradeoff).
    vector<int> blobSizes;
    vector<int> bridgeLens;             // Bc-1 entries
    vector<vector<int>> armLens;        // per-blob dead-end pendant arm lengths
    int k;

    switch (testId){
        case 2: blobSizes={10,10}; bridgeLens={8}; armLens={{},{}}; k=2; break;
        case 3: blobSizes={12,14,12}; bridgeLens={10,12}; armLens={{},{},{}}; k=4; break;
        case 4: blobSizes={40,36}; bridgeLens={140}; armLens={{},{}}; k=3; break;              // TRAP: long single bridge, spare depot
        case 5: blobSizes={16,18,15,17}; bridgeLens={18,22,16}; armLens={{},{},{},{}}; k=5; break;
        case 6: blobSizes={55,30}; bridgeLens={220}; armLens={{},{}}; k=3; break;               // TRAP: 2 blocks + one VERY long bridge, spare depot
        case 7: blobSizes={20,22,18,24,20}; bridgeLens={15,20,14,18}; armLens={{},{},{},{},{}}; k=6; break;
        case 8: blobSizes={35,32,35}; bridgeLens={130,45}; armLens={{},{},{}}; k=4; break;     // TRAP: very uneven bridge lengths, spare depot
        case 9: blobSizes={35,30,30}; bridgeLens={250,60}; armLens={{},{},{}}; k=5; break;     // TRAP: 3 blocks, one VERY long + one short bridge, 2 spare depots
        case 10: blobSizes={40,38,40,36}; bridgeLens={140,80,70}; armLens={{},{},{},{}}; k=6; break; // TRAP: largest, several long bridges, spare depots
        default: blobSizes={10,10}; bridgeLens={8}; armLens={{},{}}; k=2; break;
    }

    int prevPort = -1;
    for (size_t bi = 0; bi < blobSizes.size(); bi++){
        vector<int> ids;
        if (bi > 0){
            int len = bridgeLens[bi - 1];
            int prev = prevPort;
            for (int i = 0; i < len; i++){
                int v = nextId++;
                setPop(v, rnd.next(1, 3));
                edges.push_back({prev, v});
                prev = v;
            }
            int chainTail = prev;
            int blobFirst = nextId;
            ids = buildBlob(blobSizes[bi], 3.6);
            edges.push_back({chainTail, blobFirst});
        } else {
            ids = buildBlob(blobSizes[bi], 3.6);
        }
        for (int len : armLens[bi]){
            int port = ids[rnd.next(0, (int)ids.size() - 1)];
            buildPendant(port, len);
        }
        // pick a fresh random port within this blob for the NEXT bridge
        prevPort = ids[rnd.next(0, (int)ids.size() - 1)];
    }
    int n = nextId - 1;
    emit(n, k);
    return 0;
}
