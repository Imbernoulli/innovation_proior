// TIER: strong
// Multi-restart list scheduling. We evaluate several dispatch policies -- a dynamic
// earliest-completion rule, a WSPT static priority (weight per remaining work), and
// many seeded random static ride priorities -- and keep the schedule with the
// smallest total weighted opening time. Deterministic (fixed RNG seed).
#include <bits/stdc++.h>
using namespace std;

static int n, m, totalOps;
static vector<int> w;
static vector<vector<int>> mach, dur;

// List schedule using a static ride "rank" (lower = higher priority). Among the
// frontier tasks, pick the one whose ride has the best rank; break ties by earliest
// feasible finish. Returns weighted opening time; fills `start`.
long long scheduleByRank(const vector<int>& rank, vector<vector<long long>>& start) {
    vector<int> nextOp(n, 0);
    vector<long long> jobReady(n, 0), machFree(m, 0);
    for (int j = 0; j < n; j++) start[j].assign(mach[j].size(), 0);

    for (int done = 0; done < totalOps; done++) {
        int bj = -1;
        long long bestFin = LLONG_MAX;
        int bestRank = INT_MAX;
        for (int j = 0; j < n; j++) {
            int k = nextOp[j];
            if (k >= (int)mach[j].size()) continue;
            long long st = max(jobReady[j], machFree[mach[j][k]]);
            long long fin = st + dur[j][k];
            if (rank[j] < bestRank || (rank[j] == bestRank && fin < bestFin)) {
                bestRank = rank[j]; bestFin = fin; bj = j;
            }
        }
        int k = nextOp[bj];
        long long st = max(jobReady[bj], machFree[mach[bj][k]]);
        start[bj][k] = st;
        jobReady[bj] = st + dur[bj][k];
        machFree[mach[bj][k]] = st + dur[bj][k];
        nextOp[bj]++;
    }
    long long F = 0;
    for (int j = 0; j < n; j++)
        F += (long long)w[j] * (start[j].back() + dur[j].back());
    return F;
}

// Dynamic weighted dispatch: at each step pick the frontier task minimizing
//   key = finish_time - alpha * w[j]
// (alpha = 0 is pure earliest-completion; larger alpha favors popular rides). Returns F.
long long scheduleWeighted(double alpha, vector<vector<long long>>& start) {
    vector<int> nextOp(n, 0);
    vector<long long> jobReady(n, 0), machFree(m, 0);
    for (int j = 0; j < n; j++) start[j].assign(mach[j].size(), 0);
    for (int done = 0; done < totalOps; done++) {
        int bj = -1; double bestKey = 1e300; long long bestFin = LLONG_MAX;
        for (int j = 0; j < n; j++) {
            int k = nextOp[j];
            if (k >= (int)mach[j].size()) continue;
            long long st = max(jobReady[j], machFree[mach[j][k]]);
            long long fin = st + dur[j][k];
            double key = (double)fin - alpha * (double)w[j];
            if (key < bestKey - 1e-9 || (fabs(key - bestKey) <= 1e-9 && fin < bestFin)) {
                bestKey = key; bestFin = fin; bj = j;
            }
        }
        int k = nextOp[bj];
        long long st = max(jobReady[bj], machFree[mach[bj][k]]);
        start[bj][k] = st; jobReady[bj] = st + dur[bj][k];
        machFree[mach[bj][k]] = st + dur[bj][k]; nextOp[bj]++;
    }
    long long F = 0;
    for (int j = 0; j < n; j++) F += (long long)w[j] * (start[j].back() + dur[j].back());
    return F;
}

int main() {
    if (scanf("%d %d", &n, &m) != 2) return 0;
    w.resize(n); mach.resize(n); dur.resize(n);
    totalOps = 0;
    for (int j = 0; j < n; j++) {
        int o; scanf("%d %d", &w[j], &o);
        mach[j].resize(o); dur[j].resize(o);
        for (int k = 0; k < o; k++) scanf("%d %d", &mach[j][k], &dur[j][k]);
        totalOps += o;
    }

    vector<vector<long long>> best, cur(n);
    long long bestF = LLONG_MAX;

    auto consider = [&](long long F) {
        if (F < bestF) { bestF = F; best = cur; }
    };

    // 1) dynamic weighted dispatch over a grid of alpha (alpha=0 == pure ECT)
    for (double alpha : {0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0})
        consider(scheduleWeighted(alpha, cur));

    // 2) WSPT static: rank by remaining work / weight ascending (heavy, light-work first)
    {
        vector<long long> work(n, 0);
        for (int j = 0; j < n; j++) for (int d : dur[j]) work[j] += d;
        vector<int> idx(n); iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            // smaller work/weight => higher priority
            return (double)work[a] / w[a] < (double)work[b] / w[b];
        });
        vector<int> rank(n);
        for (int r = 0; r < n; r++) rank[idx[r]] = r;
        consider(scheduleByRank(rank, cur));
    }

    // 3) many seeded random static priorities
    mt19937 rng(987654321u);
    vector<int> perm(n); iota(perm.begin(), perm.end(), 0);
    int restarts = 400;
    for (int it = 0; it < restarts; it++) {
        shuffle(perm.begin(), perm.end(), rng);
        vector<int> rank(n);
        for (int r = 0; r < n; r++) rank[perm[r]] = r;
        consider(scheduleByRank(rank, cur));
    }

    for (int j = 0; j < n; j++) {
        string line;
        for (size_t k = 0; k < best[j].size(); k++) {
            if (k) line += ' ';
            line += to_string(best[j][k]);
        }
        printf("%s\n", line.c_str());
    }
    return 0;
}
