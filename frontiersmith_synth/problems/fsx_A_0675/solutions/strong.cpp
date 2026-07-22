// TIER: strong
// Insight: conflict cost between two actors is NOT their raw access frequency, it is how
// often their cues directly ALTERNATE in time. We mine that by building a weighted
// "temporal conflict graph" from consecutive-cue transitions in the trace (an edge (i,j)
// gets weight += 1 every time the cue sheet goes ...i,j... or ...j,i... back to back).
// Two hugely popular actors whose hot windows never touch get NO edge between them and are
// free to share a room; two only-moderately-popular actors that ping-pong constantly get a
// heavy edge and must be kept apart (or need more room capacity than is available).
//
// We then greedily color actors into rooms in order of heaviest total incident weight,
// choosing at each step the room whose already-placed occupants have the smallest total
// edge weight to this actor (a weighted-graph-coloring construction, not a frequency bin
// pack). Finally we run a time-boxed local search that mutates the assignment and
// re-simulates the ACTUAL objective (the true LRU-replay slow-change count) to escape the
// greedy construction's local optimum.
#include <bits/stdc++.h>
using namespace std;
using Clock = chrono::steady_clock;

static int P, S, K, T;
static vector<int> trace;

static long long simulate(const vector<int> &color) {
    vector<array<int, 4>> rk(S + 1);
    vector<int> sz(S + 1, 0);
    long long misses = 0;
    for (int t = 0; t < T; t++) {
        int a = trace[t];
        int r = color[a];
        int pos = -1;
        for (int i = 0; i < sz[r]; i++) if (rk[r][i] == a) { pos = i; break; }
        if (pos >= 0) {
            int v = rk[r][pos];
            for (int i = pos; i > 0; i--) rk[r][i] = rk[r][i - 1];
            rk[r][0] = v;
        } else {
            misses++;
            if (sz[r] >= K) {
                sz[r]--;
            }
            for (int i = min(sz[r], K - 1); i > 0; i--) rk[r][i] = rk[r][i - 1];
            rk[r][0] = a;
            if (sz[r] < K) sz[r]++;
        }
    }
    return misses;
}

int main() {
    auto t0 = Clock::now();
    scanf("%d %d %d %d", &P, &S, &K, &T);
    trace.resize(T);
    for (int t = 0; t < T; t++) scanf("%d", &trace[t]);

    // --- mine the weighted temporal-alternation graph from consecutive distinct cues ---
    unordered_map<long long, int> ew;
    ew.reserve(T * 2);
    const long long BASE = 2000; // P <= 1500
    for (int t = 1; t < T; t++) {
        int u = trace[t - 1], v = trace[t];
        if (u == v) continue;
        int a = min(u, v), b = max(u, v);
        ew[(long long)a * BASE + b]++;
    }

    vector<vector<pair<int, int>>> adj(P + 1);
    vector<long long> deg(P + 1, 0);
    for (auto &kv : ew) {
        int a = (int)(kv.first / BASE), b = (int)(kv.first % BASE);
        int w = kv.second;
        adj[a].push_back({b, w});
        adj[b].push_back({a, w});
        deg[a] += w;
        deg[b] += w;
    }

    vector<int> order(P);
    for (int i = 0; i < P; i++) order[i] = i + 1;
    sort(order.begin(), order.end(), [&](int x, int y) {
        if (deg[x] != deg[y]) return deg[x] > deg[y];
        return x < y;
    });

    vector<int> color(P + 1, 0);
    vector<char> placed(P + 1, 0);
    vector<long long> load(S + 1, 0);
    for (int a : order) {
        vector<long long> cost(S + 1, 0);
        for (auto &pr : adj[a]) {
            int nb = pr.first, w = pr.second;
            if (placed[nb]) cost[color[nb]] += w;
        }
        int best = 1;
        for (int r = 2; r <= S; r++) {
            if (cost[r] < cost[best] ||
                (cost[r] == cost[best] && load[r] < load[best])) {
                best = r;
            }
        }
        color[a] = best;
        placed[a] = 1;
        load[best]++;
    }

    long long bestF = simulate(color);
    vector<int> best = color;

    // --- time-boxed local search directly on the true replayed objective ---
    mt19937 rng(12345);
    double budgetMs = 2600.0;
    long long curF = bestF;
    vector<int> cur = color;
    while (true) {
        double elapsed = chrono::duration<double, milli>(Clock::now() - t0).count();
        if (elapsed > budgetMs) break;
        int a = 1 + (int)(rng() % P);
        int oldc = cur[a];
        int newc = 1 + (int)(rng() % S);
        if (newc == oldc) continue;
        cur[a] = newc;
        long long f = simulate(cur);
        if (f <= curF) {
            curF = f;
            if (f < bestF) { bestF = f; best = cur; }
        } else {
            cur[a] = oldc; // revert
        }
    }

    for (int i = 1; i <= P; i++) {
        printf("%d%c", best[i], (i < P) ? ' ' : '\n');
    }
    return 0;
}
