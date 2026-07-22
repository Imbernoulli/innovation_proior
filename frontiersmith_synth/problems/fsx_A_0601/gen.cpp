#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// -----------------------------------------------------------------------------
// Generator for "Quarantine Cuts on a Drifting Contact Graph" (eigen-drift-quarantine).
//
// Each test is a connected contact graph with C overlapping dense communities ("cores")
// plus a sparse periphery. Core 0 is the DOMINANT core (largest internal degree ->
// largest local spectral radius); cores 1..C-1 are SECONDARY cores with internal degrees
// only slightly below core 0, and near-equal to each other. This is the eigenvector-drift
// trap: cuts ranked by the ORIGINAL principal eigenvector all land in core 0; once core 0 is
// suppressed the top eigenvalue migrates to an untouched secondary core, so a myopic
// (static-ranking) budget is wasted. Balancing cuts across all cores (following the drifting
// certificate) does far better.
//
// testId is a size/adversarial ladder: testId 1 is tiny (example scale); testId 10 fills the
// envelope (n ~ 4800, m ~ 80000). Cores stay near-degenerate on every case (the trap is the
// point), with more cores / larger cores as testId grows.
// -----------------------------------------------------------------------------

struct Cfg { int C, s; vector<int> degs; int periph, conn, k; };

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    vector<Cfg> table = {
        {3, 22,  {12,8,8},                     30,   2, 80},
        {4, 40,  {22,17,16,16},                80,   3, 500},
        {5, 55,  {28,22,21,21,20},             150,  3, 1000},
        {4, 80,  {34,26,25,24},                200,  3, 1600},
        {6, 60,  {30,23,23,22,22,22},          200,  3, 1600},
        {5, 90,  {38,29,28,27,27},             250,  4, 2200},
        {7, 70,  {34,26,26,25,25,24,24},       300,  4, 2400},
        {6, 110, {44,34,33,32,32,31},          300,  4, 3200},
        {8, 140, {46,36,35,34,34,33,33,32},    400,  5, 4200},
        {8, 400, {60,47,46,45,45,44,44,43},    1600, 6, 9000},
    };
    int idx = testId - 1;
    if (idx < 0) idx = 0;
    if (idx >= (int)table.size()) idx = (int)table.size() - 1;
    Cfg cfg = table[idx];

    int C = cfg.C, s = cfg.s, periph = cfg.periph, conn = cfg.conn, k = cfg.k;

    // node id allocation: core 0 nodes, core 1 nodes, ..., then periphery nodes (0-indexed)
    int nodes = 0;
    vector<vector<int>> core(C);
    for (int i = 0; i < C; i++) {
        core[i].resize(s);
        for (int j = 0; j < s; j++) core[i][j] = nodes++;
    }
    vector<int> pn(periph);
    for (int j = 0; j < periph; j++) pn[j] = nodes++;
    int n = nodes;

    set<pair<int,int>> E;
    auto addE = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        E.insert({a, b});
    };

    // cores: an intra-core cycle (guarantees each core is internally connected) plus a
    // random d-regular-ish overlay setting the core's internal density (=> local spectral
    // radius ~ degs[i]).
    for (int i = 0; i < C; i++) {
        int d = cfg.degs[i];
        for (int j = 0; j < s; j++) addE(core[i][j], core[i][(j + 1) % s]);
        for (int j = 0; j < s; j++) {
            for (int t = 0; t < d; t++) {
                int v = core[i][rnd.next(0, s - 1)];
                addE(core[i][j], v);
            }
        }
    }

    // periphery: a spanning chain (keeps everything connected) plus a light random mesh.
    for (int j = 0; j + 1 < periph; j++) addE(pn[j], pn[j + 1]);
    for (int j = 0; j < periph; j++) {
        for (int t = 0; t < 2; t++) addE(pn[j], pn[rnd.next(0, periph - 1)]);
    }

    // connect each core to the periphery (cores overlap the backbone through these links).
    for (int i = 0; i < C; i++) {
        for (int t = 0; t < conn; t++) {
            int u = core[i][rnd.next(0, s - 1)];
            int v = pn[rnd.next(0, periph - 1)];
            addE(u, v);
        }
    }
    addE(core[0][0], pn[0]); // guarantee core 0 touches the backbone

    // emit (1-indexed), shuffled so the input carries no positional leakage.
    vector<pair<int,int>> edges(E.begin(), E.end());
    shuffle(edges.begin(), edges.end());
    int m = (int)edges.size();

    // clamp budget so a spanning tree always survives: k < m - (n-1)
    int maxk = m - (n - 1) - 1;
    if (k > maxk) k = maxk;
    if (k < 1) k = 1;

    printf("%d %d %d\n", n, m, k);
    for (auto &e : edges) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
