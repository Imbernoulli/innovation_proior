// TIER: strong
// Cost-effectiveness greedy (maximize newly-warmed sectors per unit cost) with redundancy
// pruning, run over several SEEDED randomized restarts; keep the cheapest feasible cover.
#include <bits/stdc++.h>
using namespace std;

static uint64_t rngState = 88172645463325252ULL;
static inline uint64_t xr() { rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17; return rngState; }

int n;
vector<int> cost, rad;
vector<vector<int>> ball;

// Build a feasible cover using cost-effectiveness greedy with a small random perturbation
// on ties (eps in [0,1)); returns chosen set. jitter scales the tie perturbation.
vector<int> buildGreedy(double jitter) {
    vector<char> warm(n + 1, 0);
    int coldLeft = n;
    vector<int> chosen;
    while (coldLeft > 0) {
        int best = -1; double bestScore = -1;
        for (int s = 1; s <= n; s++) {
            int nw = 0;
            for (int c : ball[s]) if (!warm[c]) nw++;
            if (nw <= 0) continue;
            // effectiveness = newly warmed per unit cost, with tiny random tie perturbation
            double eps = 1.0 + jitter * ((double)(xr() % 1000) / 1000.0);
            double sc = ((double)nw / (double)cost[s]) * eps;
            if (sc > bestScore) { bestScore = sc; best = s; }
        }
        if (best < 0) break;
        chosen.push_back(best);
        for (int c : ball[best]) if (!warm[c]) { warm[c] = 1; coldLeft--; }
    }
    return chosen;
}

// Remove relays whose whole ball is still covered by the rest (most expensive first).
void prune(vector<int>& chosen) {
    vector<int> cover(n + 1, 0);
    for (int s : chosen) for (int c : ball[s]) cover[c]++;
    // sort candidate removals by descending cost (drop expensive redundant relays first)
    sort(chosen.begin(), chosen.end(), [&](int a, int b){ return cost[a] > cost[b]; });
    vector<int> keep;
    // process in the sorted order; a relay is removable iff every cell it warms has cover>1
    for (int s : chosen) {
        bool redundant = true;
        for (int c : ball[s]) if (cover[c] <= 1) { redundant = false; break; }
        if (redundant) { for (int c : ball[s]) cover[c]--; }
        else keep.push_back(s);
    }
    chosen = keep;
}

long long totalCost(const vector<int>& chosen) {
    long long f = 0; for (int s : chosen) f += cost[s]; return f;
}

int main() {
    int H, W;
    if (scanf("%d %d", &H, &W) != 2) return 0;
    vector<string> grid(H);
    for (int i = 0; i < H; i++) { char buf[64]; scanf("%s", buf); grid[i] = buf; }
    scanf("%d", &n);
    cost.assign(n + 1, 0); rad.assign(n + 1, 0);
    for (int v = 1; v <= n; v++) scanf("%d", &cost[v]);
    for (int v = 1; v <= n; v++) scanf("%d", &rad[v]);

    vector<vector<int>> id(H, vector<int>(W, -1));
    int m = 0;
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++)
            if (grid[i][j] == '.') id[i][j] = ++m;
    vector<vector<int>> adj(n + 1);
    const int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            if (id[i][j] < 0) continue;
            for (int d = 0; d < 4; d++) {
                int ni = i + dx[d], nj = j + dy[d];
                if (ni < 0 || ni >= H || nj < 0 || nj >= W) continue;
                if (id[ni][nj] < 0) continue;
                adj[id[i][j]].push_back(id[ni][nj]);
            }
        }

    ball.assign(n + 1, {});
    vector<int> dist(n + 1, -1), stamp(n + 1, -1);
    for (int s = 1; s <= n; s++) {
        queue<int> q; dist[s] = 0; stamp[s] = s; q.push(s);
        while (!q.empty()) {
            int x = q.front(); q.pop();
            ball[s].push_back(x);
            if (dist[x] == rad[s]) continue;
            for (int y : adj[x]) if (stamp[y] != s) { stamp[y] = s; dist[y] = dist[x] + 1; q.push(y); }
        }
    }

    // deterministic multi-start: first run with no jitter, then jittered restarts
    vector<int> bestSet;
    long long bestF = LLONG_MAX;
    int RESTARTS = 24;
    for (int t = 0; t < RESTARTS; t++) {
        double jitter = (t == 0) ? 0.0 : 0.35;
        vector<int> cand = buildGreedy(jitter);
        prune(cand);
        long long f = totalCost(cand);
        if (f < bestF) { bestF = f; bestSet = cand; }
    }

    printf("%d\n", (int)bestSet.size());
    for (size_t i = 0; i < bestSet.size(); i++)
        printf("%d%c", bestSet[i], i + 1 == bestSet.size() ? '\n' : ' ');
    if (bestSet.empty()) printf("\n");
    return 0;
}
