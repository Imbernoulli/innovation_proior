#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    registerGen(argc, argv, 1);
    int testId = atoi(argv[1]);

    // ---- scale ladder: vineyard junction count grows with testId ----
    // testId 1 tiny (example scale), up to a large field by testId 10.
    int n = 8 + (testId - 1) * 160;          // 8, 168, ..., 1448
    const int COORD = 10000;
    const int D = 3;                          // max valve fan-out (degree bound)

    // random junction coordinates in the field
    vector<long long> px(n), py(n);
    for (int i = 0; i < n; i++) {
        px[i] = rnd.next(0, COORD);
        py[i] = rnd.next(0, COORD);
    }

    // ---- backbone ordering: boustrophedon strip sweep (a decent-but-not-great
    //      Hamiltonian path). Node input indices follow this order, so consecutive
    //      indices (i,i+1) form a guaranteed feasible spanning path (the baseline).
    int s = max(1, (int)llround(sqrt((double)n)));
    vector<int> byx(n);
    iota(byx.begin(), byx.end(), 0);
    sort(byx.begin(), byx.end(), [&](int a, int b){
        if (px[a] != px[b]) return px[a] < px[b];
        return py[a] < py[b];
    });
    // assign each point to a strip by x-rank
    vector<vector<int>> strip(s);
    for (int r = 0; r < n; r++) {
        int st = (int)((long long)r * s / n);
        if (st >= s) st = s - 1;
        strip[st].push_back(byx[r]);
    }
    vector<int> order;
    for (int st = 0; st < s; st++) {
        auto& col = strip[st];
        sort(col.begin(), col.end(), [&](int a, int b){
            if (py[a] != py[b]) return py[a] < py[b];
            return px[a] < px[b];
        });
        if (st & 1) reverse(col.begin(), col.end());
        for (int idx : col) order.push_back(idx);
    }
    // relabel: node k (1-based) -> original point order[k-1]
    vector<long long> qx(n + 1), qy(n + 1);
    for (int k = 1; k <= n; k++) {
        qx[k] = px[order[k - 1]];
        qy[k] = py[order[k - 1]];
    }

    auto wdist = [&](int a, int b) -> long long {
        long long dx = qx[a] - qx[b], dy = qy[a] - qy[b];
        long long w = (long long)llround(sqrt((double)(dx * dx + dy * dy)));
        return max(1LL, w);
    };

    // ---- build candidate pipe set: backbone + K nearest neighbours ----
    set<pair<int,int>> pairs;
    for (int i = 1; i < n; i++) pairs.insert({i, i + 1});   // guaranteed backbone

    int K = 6;
    // brute-force KNN (n<=1448 -> fine)
    for (int i = 1; i <= n; i++) {
        vector<pair<long long,int>> cand;
        cand.reserve(n);
        for (int j = 1; j <= n; j++) if (j != i) cand.push_back({wdist(i, j), j});
        int kk = min((int)cand.size(), K);
        nth_element(cand.begin(), cand.begin() + kk, cand.end());
        for (int t = 0; t < kk; t++) {
            int j = cand[t].second;
            pairs.insert({min(i, j), max(i, j)});
        }
    }

    // materialize edges with weights
    vector<array<long long,3>> edges; // u, v, w
    edges.reserve(pairs.size());
    for (auto& p : pairs) edges.push_back({(long long)p.first, (long long)p.second, wdist(p.first, p.second)});

    // shuffle edge order so index != structural position
    shuffle(edges.begin(), edges.end());

    int m = (int)edges.size();
    printf("%d %d %d\n", n, m, D);
    for (auto& e : edges) printf("%lld %lld %lld\n", e[0], e[1], e[2]);
    return 0;
}
