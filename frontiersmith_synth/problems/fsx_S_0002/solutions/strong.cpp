// TIER: strong
// Multi-start randomized cost-effectiveness greedy + redundancy pruning.
// Over many seeded restarts we randomize the choice among near-best candidates, then drop
// ladders rendered redundant by later picks, keeping the cheapest feasible cover found.
#include <bits/stdc++.h>
using namespace std;

int N, M, D;
vector<vector<int>> adj;
vector<long long> c;
vector<int> r, dem;
vector<int> demIdx;
vector<vector<int>> ballDem;

// deterministic RNG
static uint64_t rngState = 0x243F6A8885A308D3ULL;
static inline uint64_t xrng() {
    rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17;
    return rngState;
}
static inline double urand() { return (double)(xrng() >> 11) / (double)(1ULL << 53); }

void computeBalls() {
    ballDem.assign(N + 1, {});
    vector<int> dist(N + 1, -1), stamp(N + 1, -1);
    for (int s = 1; s <= N; s++) {
        queue<int> q; dist[s] = 0; stamp[s] = s; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            if (demIdx[x] >= 0) ballDem[s].push_back(demIdx[x]);
            if (dist[x] == r[s]) continue;
            for (int y : adj[x]) if (stamp[y] != s) { stamp[y] = s; dist[y] = dist[x] + 1; q.push(y); }
        }
    }
}

long long totalCost(const vector<int>& ch) {
    long long f = 0; for (int v : ch) f += c[v]; return f;
}

// one randomized greedy build; frac in (0,1] controls how loose the near-best window is
vector<int> buildOnce(double frac) {
    vector<char> covered(D, 0);
    int remaining = D;
    vector<char> built(N + 1, 0);
    vector<int> chosen;
    vector<int> cand; cand.reserve(N);
    while (remaining > 0) {
        double bestEff = -1.0;
        for (int v = 1; v <= N; v++) {
            if (built[v]) continue;
            int nc = 0;
            for (int di : ballDem[v]) if (!covered[di]) nc++;
            if (nc == 0) continue;
            double eff = (double)nc / (double)c[v];
            if (eff > bestEff) bestEff = eff;
        }
        if (bestEff < 0) break;
        double thr = bestEff * frac;
        cand.clear();
        for (int v = 1; v <= N; v++) {
            if (built[v]) continue;
            int nc = 0;
            for (int di : ballDem[v]) if (!covered[di]) nc++;
            if (nc == 0) continue;
            double eff = (double)nc / (double)c[v];
            if (eff >= thr - 1e-12) cand.push_back(v);
        }
        int pick = cand[(size_t)(urand() * cand.size()) % cand.size()];
        built[pick] = 1; chosen.push_back(pick);
        for (int di : ballDem[pick]) if (!covered[di]) { covered[di] = 1; remaining--; }
    }
    return chosen;
}

// drop ladders that are redundant (every demand they cover is covered by others).
// process in order of descending cost so we shed the most expensive redundant ladders first.
vector<int> prune(vector<int> ch) {
    vector<int> cnt(D, 0);
    for (int v : ch) for (int di : ballDem[v]) cnt[di]++;
    sort(ch.begin(), ch.end(), [&](int a, int b){ return c[a] > c[b]; });
    vector<int> keep;
    for (int v : ch) {
        bool redundant = true;
        for (int di : ballDem[v]) if (cnt[di] <= 1) { redundant = false; break; }
        if (redundant) { for (int di : ballDem[v]) cnt[di]--; }
        else keep.push_back(v);
    }
    return keep;
}

int main() {
    scanf("%d %d", &N, &M);
    adj.assign(N + 1, {});
    for (int i = 0; i < M; i++) { int u, v; scanf("%d %d", &u, &v); adj[u].push_back(v); adj[v].push_back(u); }
    c.assign(N + 1, 0); r.assign(N + 1, 0);
    for (int i = 1; i <= N; i++) { int x; scanf("%d", &x); c[i] = x; }
    for (int i = 1; i <= N; i++) scanf("%d", &r[i]);
    scanf("%d", &D);
    dem.assign(D, 0); demIdx.assign(N + 1, -1);
    for (int i = 0; i < D; i++) { scanf("%d", &dem[i]); demIdx[dem[i]] = i; }

    computeBalls();

    long long bestCost = LLONG_MAX;
    vector<int> bestSet;
    const int RESTARTS = 60;
    for (int t = 0; t < RESTARTS; t++) {
        double frac = (t == 0) ? 1.0 : (0.55 + 0.45 * urand());  // t=0 = pure greedy
        vector<int> ch = buildOnce(frac);
        ch = prune(ch);
        long long f = totalCost(ch);
        if (f < bestCost) { bestCost = f; bestSet = ch; }
    }

    printf("%d\n", (int)bestSet.size());
    for (size_t i = 0; i < bestSet.size(); i++)
        printf("%d%c", bestSet[i], i + 1 == bestSet.size() ? '\n' : ' ');
    if (bestSet.empty()) printf("\n");
    return 0;
}
