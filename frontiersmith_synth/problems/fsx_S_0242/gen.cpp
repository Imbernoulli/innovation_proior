#include "testlib.h"
#include <bits/stdc++.h>
using namespace std;

// Smart-City Adaptive Lighting: minimum-cost R-multi-cover generator.
// testId 1..10 is a difficulty ladder: tiny path (example scale) -> large,
// dense, cost-skewed, high-demand adversarial instances.

int main(int argc, char* argv[]) {
    registerGen(argc, argv, 1);
    int t = atoi(argv[1]);

    // Per-test parameters.
    int N, targetM, R, demCap, costMode, structMode;
    // costMode: 0 uniform 1..20, 1 skewed (mostly expensive, few cheap).
    // structMode: 0 tree+random-extra, 1 grid+random-extra.
    switch (t) {
        case 1:  N=6;    R=1; demCap=2; targetM=4;    costMode=0; structMode=0; break;
        case 2:  N=40;   R=1; demCap=2; targetM=90;   costMode=0; structMode=0; break;
        case 3:  N=120;  R=2; demCap=2; targetM=300;  costMode=1; structMode=0; break;
        case 4:  N=300;  R=1; demCap=1; targetM=560;  costMode=0; structMode=1; break;
        case 5:  N=500;  R=2; demCap=3; targetM=1400; costMode=0; structMode=0; break;
        case 6:  N=800;  R=1; demCap=2; targetM=2100; costMode=1; structMode=0; break;
        case 7:  N=1000; R=2; demCap=2; targetM=1900; costMode=0; structMode=1; break;
        case 8:  N=1200; R=1; demCap=3; targetM=3600; costMode=1; structMode=0; break;
        case 9:  N=1600; R=2; demCap=3; targetM=4600; costMode=1; structMode=0; break;
        case 10: N=2000; R=2; demCap=4; targetM=6000; costMode=1; structMode=0; break;
        default: N=2000; R=2; demCap=4; targetM=6000; costMode=1; structMode=0; break;
    }

    set<pair<int,int>> es;
    auto addEdge = [&](int a, int b) {
        if (a == b) return;
        if (a > b) swap(a, b);
        es.insert({a, b});
    };

    if (structMode == 1) {
        // Grid: pick rows x cols close to N.
        int rows = max(2, (int)floor(sqrt((double)N)));
        int cols = (N + rows - 1) / rows;
        N = rows * cols;
        auto id = [&](int r, int c) { return r * cols + c + 1; };
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                if (c + 1 < cols) addEdge(id(r, c), id(r, c + 1));
                if (r + 1 < rows) addEdge(id(r, c), id(r + 1, c));
            }
    } else {
        // Random spanning tree (connected) via random attach.
        for (int i = 2; i <= N; i++) {
            int p = rnd.next(1, i - 1);
            addEdge(p, i);
        }
    }
    // Add random extra edges up to targetM (capped).
    long long maxE = (long long)N * (N - 1) / 2;
    int cap = (int)min<long long>(targetM, maxE);
    int guard = 0;
    while ((int)es.size() < cap && guard < cap * 20 + 1000) {
        guard++;
        int a = rnd.next(1, N), b = rnd.next(1, N);
        addEdge(a, b);
    }
    int M = (int)es.size();

    // Build adjacency for demand-cap computation.
    vector<vector<int>> adj(N + 1);
    for (auto& e : es) { adj[e.first].push_back(e.second); adj[e.second].push_back(e.first); }

    // Ball size (number of vertices within R hops, inclusive) via BFS depth R.
    vector<int> ballSize(N + 1, 1);
    {
        vector<int> stamp(N + 1, 0), dist(N + 1, 0);
        int tok = 0;
        for (int s = 1; s <= N; s++) {
            tok++;
            int cnt = 0;
            queue<int> q; q.push(s); stamp[s] = tok; dist[s] = 0; cnt++;
            while (!q.empty()) {
                int x = q.front(); q.pop();
                if (dist[x] == R) continue;
                for (int y : adj[x]) if (stamp[y] != tok) {
                    stamp[y] = tok; dist[y] = dist[x] + 1; cnt++; q.push(y);
                }
            }
            ballSize[s] = cnt;
        }
    }

    // Costs.
    vector<int> c(N + 1);
    for (int i = 1; i <= N; i++) {
        if (costMode == 0) c[i] = rnd.next(1, 20);
        else {
            // skewed: ~15% cheap (1..3), rest expensive (12..20).
            if (rnd.next(0, 99) < 15) c[i] = rnd.next(1, 3);
            else c[i] = rnd.next(12, 20);
        }
    }

    // Demands: 1..min(demCap, ballSize).
    vector<int> d(N + 1);
    for (int i = 1; i <= N; i++) {
        int hi = min(demCap, ballSize[i]);
        if (hi < 1) hi = 1;
        d[i] = rnd.next(1, hi);
    }

    // Emit.
    printf("%d %d %d\n", N, M, R);
    for (int i = 1; i <= N; i++) printf("%d%c", c[i], i == N ? '\n' : ' ');
    for (int i = 1; i <= N; i++) printf("%d%c", d[i], i == N ? '\n' : ' ');
    vector<pair<int,int>> el(es.begin(), es.end());
    // shuffle edge order for realism.
    shuffle(el.begin(), el.end());
    for (auto& e : el) printf("%d %d\n", e.first, e.second);
    return 0;
}
