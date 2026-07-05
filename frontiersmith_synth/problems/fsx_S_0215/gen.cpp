#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- structure ladder ----
    // pandemic contact net: households/workplaces are dense near-cliques,
    // wired together by sparse cross-cluster contacts, plus a few high-value
    // high-degree super-connector individuals. testId 1 is tiny, grows dense/large.
    int C = 3 + testId;                        // number of clusters: 4 .. 13
    double pIntra = 0.72 + 0.02 * (testId % 4);// near-clique intra density
    int wLoBig = 200 + 10 * (testId % 5);      // super-connector weight floor
    int wHiBig = 400;                          // super-connector weight ceiling

    vector<int> w;                             // 0-based vertex weights
    vector<vector<int>> cluster;               // vertex ids per cluster
    int cur = 0;
    for (int c = 0; c < C; c++) {
        int sz = rnd.next(6, 6 + 11 * testId); // cluster size grows with testId
        vector<int> nodes;
        for (int i = 0; i < sz; i++) {
            w.push_back(rnd.next(1, 100));
            nodes.push_back(cur++);
        }
        cluster.push_back(nodes);
    }
    int nComm = cur;

    // super-connector (hub) vertices: high value, high degree
    int nHubs = 1 + testId / 2;                // 1 .. 6
    vector<int> hubs;
    for (int h = 0; h < nHubs; h++) {
        w.push_back(rnd.next(wLoBig, wHiBig));
        hubs.push_back(cur++);
    }
    int n = cur;

    set<pair<int,int>> E;
    auto addE = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        E.insert({a, b});
    };

    // dense intra-cluster contacts (near-cliques)
    for (auto& nodes : cluster) {
        int sz = (int)nodes.size();
        for (int i = 0; i < sz; i++)
            for (int j = i + 1; j < sz; j++)
                if (rnd.next(0.0, 1.0) < pIntra) addE(nodes[i], nodes[j]);
    }

    // sparse cross-cluster contacts (this is what makes it a GENERAL graph)
    int interE = 2 * nComm + 3 * testId;
    for (int e = 0; e < interE; e++) {
        int a = rnd.next(0, nComm - 1), b = rnd.next(0, nComm - 1);
        addE(a, b);
    }

    // super-connectors touch a large random slice of the population
    for (int h : hubs) {
        int deg = max(4, (int)(nComm * (0.30 + 0.05 * (testId % 3))));
        for (int d = 0; d < deg; d++) addE(h, rnd.next(0, nComm - 1));
    }
    for (int i = 0; i < (int)hubs.size(); i++)
        for (int j = i + 1; j < (int)hubs.size(); j++)
            addE(hubs[i], hubs[j]);

    // hide structure: random relabeling of vertex ids
    vector<int> perm(n);
    for (int i = 0; i < n; i++) perm[i] = i;
    shuffle(perm.begin(), perm.end());
    vector<int> nw(n);
    for (int i = 0; i < n; i++) nw[perm[i]] = w[i];

    vector<pair<int,int>> edges;
    edges.reserve(E.size());
    for (auto& pr : E) edges.push_back({perm[pr.first], perm[pr.second]});
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d\n", n, m);
    for (int i = 0; i < n; i++) printf("%d%c", nw[i], i + 1 < n ? ' ' : '\n');
    for (auto& e : edges) printf("%d %d\n", e.first + 1, e.second + 1);
    return 0;
}
