// TIER: strong
// Multi-restart randomized non-delay list scheduling for total weighted completion.
// Each restart uses a static corridor-priority key to break ties among ready segments
// that share the same earliest feasible start; we try WSPT-style keys plus many seeded
// random keys and keep the schedule with the smallest weighted completion.
#include <bits/stdc++.h>
using namespace std;

int n, m, total;
vector<int> o;
vector<long long> w;
vector<vector<int>> mach;
vector<vector<long long>> dur;
vector<long long> remWork;   // total remaining work per corridor

// Run non-delay list scheduling with corridor tie-break key `key` (smaller = earlier).
// Returns weighted completion F and fills `start`.
long long schedule(const vector<double>& key, vector<vector<long long>>& start) {
    vector<long long> machFree(m, 0), jobAvail(n, 0);
    vector<int> nxt(n, 0);
    for (int j = 0; j < n; j++) start[j].assign(o[j], 0);

    for (int done = 0; done < total; done++) {
        long long bestStart = LLONG_MAX;
        double bestKey = 0;
        int bj = -1;
        for (int j = 0; j < n; j++) {
            if (nxt[j] >= o[j]) continue;
            int k = nxt[j];
            long long st = max(jobAvail[j], machFree[mach[j][k]]);
            if (st < bestStart || (st == bestStart && key[j] < bestKey)) {
                bestStart = st; bestKey = key[j]; bj = j;
            }
        }
        int k = nxt[bj];
        start[bj][k] = bestStart;
        long long e = bestStart + dur[bj][k];
        machFree[mach[bj][k]] = e;
        jobAvail[bj] = e;
        nxt[bj]++;
    }
    long long F = 0;
    for (int j = 0; j < n; j++) F += w[j] * (start[j][o[j]-1] + dur[j][o[j]-1]);
    return F;
}

int main() {
    scanf("%d %d", &n, &m);
    o.resize(n); w.resize(n); mach.resize(n); dur.resize(n); remWork.assign(n, 0);
    total = 0;
    for (int j = 0; j < n; j++) {
        scanf("%d %lld", &o[j], &w[j]);
        mach[j].resize(o[j]); dur[j].resize(o[j]);
        for (int k = 0; k < o[j]; k++) {
            scanf("%d %lld", &mach[j][k], &dur[j][k]);
            remWork[j] += dur[j][k];
        }
        total += o[j];
    }

    vector<vector<long long>> best, cur(n);
    long long bestF = LLONG_MAX;

    auto tryKey = [&](const vector<double>& key) {
        long long F = schedule(key, cur);
        if (F < bestF) { bestF = F; best = cur; }
    };

    // Deterministic structured keys.
    vector<double> k1(n), k2(n), k3(n);
    for (int j = 0; j < n; j++) {
        k1[j] = (double)remWork[j] / (double)w[j];   // WSPT: urgent-short first
        k2[j] = -(double)w[j];                        // heaviest weight first
        k3[j] = (double)remWork[j];                   // shortest remaining first
    }
    tryKey(k1); tryKey(k2); tryKey(k3);

    // Seeded random restarts: perturb the WSPT key.
    mt19937_64 rng(0x5eed1234abcdULL);
    int restarts = 220;
    for (int r = 0; r < restarts; r++) {
        vector<double> key(n);
        for (int j = 0; j < n; j++) {
            double noise = (double)(rng() % 2000) / 1000.0 - 1.0;  // [-1,1)
            key[j] = k1[j] * (1.0 + 0.7 * noise);
        }
        tryKey(key);
    }

    for (int j = 0; j < n; j++)
        for (int k = 0; k < o[j]; k++)
            printf("%lld%c", best[j][k], k + 1 < o[j] ? ' ' : '\n');
    return 0;
}
