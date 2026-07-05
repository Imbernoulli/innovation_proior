// TIER: strong
// Randomized multi-restart local search. From several seeded inits (greedy + random),
// repeatedly move each hydrophone to the channel minimizing its local interference given
// ALL current neighbors, until a fixed point; keep the assignment with lowest total F.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int n, m, K;
vector<vector<array<int,3>>> adj; // (nbr, w, d)

ll totalF(const vector<int>& c) {
    ll F = 0;
    for (int i = 1; i <= n; i++)
        for (auto &e : adj[i]) {
            if (e[0] < i) continue; // count each undirected pair once
            int diff = abs(c[i] - c[e[0]]);
            int pen = e[2] - diff;
            if (pen > 0) F += (ll)e[1] * pen;
        }
    return F;
}

// best channel for node i given current neighbor channels; returns (cost, ch)
pair<ll,int> bestChannel(int i, const vector<int>& c) {
    ll best = LLONG_MAX; int bestCh = c[i];
    for (int ch = 1; ch <= K; ch++) {
        ll cost = 0;
        for (auto &e : adj[i]) {
            int diff = abs(ch - c[e[0]]);
            int pen = e[2] - diff;
            if (pen > 0) cost += (ll)e[1] * pen;
        }
        if (cost < best) { best = cost; bestCh = ch; }
    }
    return {best, bestCh};
}

void localSearch(vector<int>& c) {
    int passes = 0;
    bool improved = true;
    while (improved && passes < 60) {
        improved = false; passes++;
        for (int i = 1; i <= n; i++) {
            auto pr = bestChannel(i, c);
            if (pr.second != c[i]) { c[i] = pr.second; improved = true; }
        }
    }
}

int main() {
    scanf("%d %d %d", &n, &m, &K);
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v, w, d;
        scanf("%d %d %d %d", &u, &v, &w, &d);
        adj[u].push_back({v, w, d});
        adj[v].push_back({u, w, d});
    }

    // init 1: index-order greedy
    vector<int> greedy(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        ll best = LLONG_MAX; int bestCh = 1;
        for (int ch = 1; ch <= K; ch++) {
            ll cost = 0;
            for (auto &e : adj[i]) {
                if (greedy[e[0]] == 0) continue;
                int diff = abs(ch - greedy[e[0]]);
                int pen = e[2] - diff;
                if (pen > 0) cost += (ll)e[1] * pen;
            }
            if (cost < best) { best = cost; bestCh = ch; }
        }
        greedy[i] = bestCh;
    }

    vector<int> bestSol = greedy;
    localSearch(bestSol);
    ll bestVal = totalF(bestSol);

    // deterministic RNG for reproducibility
    std::mt19937 rng(987654321u);
    int restarts = 8;
    for (int r = 0; r < restarts; r++) {
        vector<int> c(n + 1);
        if (r == 0) {
            c = greedy; // re-run LS from greedy already done; use random from here
            for (int i = 1; i <= n; i++) c[i] = (int)(rng() % K) + 1;
        } else {
            for (int i = 1; i <= n; i++) c[i] = (int)(rng() % K) + 1;
        }
        localSearch(c);
        ll v = totalF(c);
        if (v < bestVal) { bestVal = v; bestSol = c; }
    }

    for (int i = 1; i <= n; i++) printf("%d%c", bestSol[i], i == n ? '\n' : ' ');
    return 0;
}
