// TIER: strong
// Multi-rule dispatch: generate non-delay schedules with several priority rules
// (EST, SPT, LPT, MWKR, LWKR, FCFS) plus many randomized tie-break restarts, and keep
// the schedule with the smallest makespan. All generated schedules are feasible.
#include <bits/stdc++.h>
using namespace std;

static int J, M, totalOps;
static vector<vector<int>> mach, dur;
static vector<vector<int>> remWork; // remWork[i][k] = sum dur[i][k..end]

static uint64_t rngState = 88172645463325252ULL;
static inline uint64_t xr() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState;
}

// rule: 0 EST,1 SPT,2 LPT,3 MWKR,4 LWKR,5 FCFS,6 RANDOM tie EST
static long long run(int rule, vector<vector<long long>> &start) {
    vector<int> nextOp(J, 0);
    vector<long long> jobReady(J, 0);
    vector<long long> bayFree(M + 1, 0);
    for (int i = 0; i < J; i++) start[i].assign(mach[i].size(), 0);

    for (int done = 0; done < totalOps; done++) {
        int best = -1;
        long long bestKey = 0, bestTieS = 0;
        for (int i = 0; i < J; i++) {
            int k = nextOp[i];
            if (k >= (int)mach[i].size()) continue;
            int m = mach[i][k];
            long long s = max(jobReady[i], bayFree[m]);
            long long d = dur[i][k];
            long long key;
            // higher key == preferred
            switch (rule) {
                case 0: key = -s; break;                 // EST: smallest s
                case 1: key = -d; break;                 // SPT
                case 2: key = d; break;                  // LPT
                case 3: key = remWork[i][k]; break;      // MWKR
                case 4: key = -remWork[i][k]; break;     // LWKR
                case 5: key = -(long long)i; break;      // FCFS
                default: key = -s + (long long)(xr() % 7); break; // random near-EST
            }
            if (best == -1 || key > bestKey || (key == bestKey && s < bestTieS)) {
                bestKey = key; bestTieS = s; best = i;
            }
        }
        int i = best, k = nextOp[i], m = mach[i][k];
        long long s = max(jobReady[i], bayFree[m]);
        start[i][k] = s;
        bayFree[m] = s + dur[i][k];
        jobReady[i] = s + dur[i][k];
        nextOp[i]++;
    }

    long long F = 0;
    for (int i = 0; i < J; i++)
        for (size_t k = 0; k < start[i].size(); k++)
            F = max(F, start[i][k] + dur[i][k]);
    return F;
}

int main() {
    if (scanf("%d %d", &J, &M) != 2) return 0;
    mach.resize(J); dur.resize(J); remWork.resize(J);
    totalOps = 0;
    for (int i = 0; i < J; i++) {
        int L; scanf("%d", &L);
        mach[i].resize(L); dur[i].resize(L); remWork[i].resize(L + 1);
        for (int k = 0; k < L; k++) scanf("%d %d", &mach[i][k], &dur[i][k]);
        remWork[i][L] = 0;
        for (int k = L - 1; k >= 0; k--) remWork[i][k] = remWork[i][k + 1] + dur[i][k];
        totalOps += L;
    }

    vector<vector<long long>> best, cur(J);
    long long bestF = LLONG_MAX;

    for (int rule = 0; rule <= 5; rule++) {
        long long F = run(rule, cur);
        if (F < bestF) { bestF = F; best = cur; }
    }
    // randomized restarts (budget scales down with size)
    int restarts = 400;
    if (totalOps > 200) restarts = 150;
    for (int r = 0; r < restarts; r++) {
        long long F = run(6, cur);
        if (F < bestF) { bestF = F; best = cur; }
    }

    for (int i = 0; i < J; i++)
        for (size_t k = 0; k < best[i].size(); k++)
            printf("%lld%c", best[i][k], k + 1 == best[i].size() ? '\n' : ' ');
    return 0;
}
