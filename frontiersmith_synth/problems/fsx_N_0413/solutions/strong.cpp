// TIER: strong
// Convex-cost successive-shortest-path routing.  For each demand we repeatedly find the path
// whose MARGINAL crowding cost a_e*(2*load_e+1) is smallest (Dijkstra over residual segments),
// then push the largest batch of passengers that is still cheaper than the inconvenience
// penalty P, using the exact quadratic increment a_e*((load+q)^2 - load^2).  This spreads load
// across parallel routes (halving the a_e*x^2 crowding vs a single route) and stops serving a
// demand once further routing costs more than leaving passengers unserved.  Beats the naive
// single-route greedy, with per-test behaviour that differs (spreading vs concentration).
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M, D; long long P;
    if (scanf("%d %d %d %lld", &N, &M, &D, &P) != 4) return 0;

    vector<int> eu(M), ev(M);
    vector<long long> ecap(M), ea(M), eload(M, 0);
    vector<vector<pair<int,int>>> adj(N + 1);  // (neighbor, edgeId)
    for (int e = 0; e < M; e++) {
        scanf("%d %d %lld %lld", &eu[e], &ev[e], &ecap[e], &ea[e]);
        adj[eu[e]].push_back({ev[e], e});
        adj[ev[e]].push_back({eu[e], e});
    }
    vector<int> sd(D), td(D);
    vector<long long> vol(D);
    for (int d = 0; d < D; d++) scanf("%d %d %lld", &sd[d], &td[d], &vol[d]);

    const long long INF = LLONG_MAX / 4;
    vector<long long> dist(N + 1);
    vector<int> parEdge(N + 1), parNode(N + 1);

    string out;
    long long K = 0;
    const int MAX_ITERS_PER_DEMAND = 8;

    for (int d = 0; d < D; d++) {
        int s = sd[d], t = td[d];
        long long remaining = vol[d];
        for (int iter = 0; iter < MAX_ITERS_PER_DEMAND && remaining > 0; iter++) {
            // Dijkstra with marginal weights over residual segments
            fill(dist.begin(), dist.end(), INF);
            fill(parEdge.begin(), parEdge.end(), -1);
            fill(parNode.begin(), parNode.end(), -1);
            dist[s] = 0;
            priority_queue<pair<long long,int>, vector<pair<long long,int>>,
                           greater<pair<long long,int>>> pq;
            pq.push({0, s});
            while (!pq.empty()) {
                auto [du, x] = pq.top(); pq.pop();
                if (du > dist[x]) continue;
                if (x == t) break;
                for (auto [y, e] : adj[x]) {
                    if (ecap[e] - eload[e] <= 0) continue;  // no residual capacity
                    long long w = ea[e] * (2 * eload[e] + 1);  // marginal for +1 unit
                    if (du + w < dist[y]) {
                        dist[y] = du + w;
                        parEdge[y] = e; parNode[y] = x;
                        pq.push({dist[y], y});
                    }
                }
            }
            if (dist[t] >= INF) break;  // no residual path
            if (dist[t] >= P) break;    // even one more passenger costs more than the penalty

            // reconstruct path, its min residual capacity
            vector<int> path, pedges;
            long long minRes = LLONG_MAX;
            int cur = t;
            while (cur != s) {
                int e = parEdge[cur];
                pedges.push_back(e);
                minRes = min(minRes, ecap[e] - eload[e]);
                path.push_back(cur);
                cur = parNode[cur];
            }
            path.push_back(s);
            reverse(path.begin(), path.end());
            reverse(pedges.begin(), pedges.end());

            // choose the largest beneficial batch q (exact quadratic increment)
            long long q = min(remaining, minRes);
            while (q > 0) {
                long long inc = 0;
                for (int e : pedges) {
                    long long lo = eload[e], hi = lo + q;
                    inc += ea[e] * (hi * hi - lo * lo);
                }
                if (inc < P * q) break;  // beneficial
                q /= 2;
            }
            if (q <= 0) break;

            for (int e : pedges) eload[e] += q;
            remaining -= q;

            out += to_string(d + 1);
            out += ' ';
            out += to_string((int)path.size());
            for (int v : path) { out += ' '; out += to_string(v); }
            out += ' ';
            out += to_string(q);
            out += '\n';
            K++;
        }
    }

    printf("%lld\n", K);
    fputs(out.c_str(), stdout);
    return 0;
}
