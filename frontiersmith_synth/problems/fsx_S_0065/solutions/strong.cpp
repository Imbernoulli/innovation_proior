// TIER: strong
// Multi-rule best-of dispatch: run several priority rules (SPT, LPT,
// most-work-remaining, earliest-completion-time) plus randomized restarts,
// each producing a full list schedule, and keep the one with the smallest
// makespan. Different rules pack the heavy-tailed "jam" operations
// differently, so this beats the single-rule greedy and varies per test.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int J, M;
vector<vector<int>> bay;
vector<vector<ll>> dur;

mt19937 rng(20240701u);

// rule: 0=SPT,1=LPT,2=MWKR,3=EFT,4=random
ll buildSchedule(int rule, vector<vector<ll>>& outSt) {
    vector<int> nxt(J, 0);
    vector<ll> jobReady(J, 0), bayFree(M + 1, 0);
    // remaining work per job (for MWKR)
    vector<ll> rem(J, 0);
    for (int j = 0; j < J; j++) for (int o = 0; o < M; o++) rem[j] += dur[j][o];

    outSt.assign(J, vector<ll>(M, 0));
    long long totalOps = (long long)J * M;
    for (long long done = 0; done < totalOps; done++) {
        int bj = -1;
        ll bkey = 0; bool first = true;
        // for random rule, collect candidates
        static vector<int> cand; cand.clear();
        for (int j = 0; j < J; j++) {
            if (nxt[j] >= M) continue;
            int o = nxt[j], m = bay[j][o];
            ll key;
            switch (rule) {
                case 0: key = dur[j][o]; break;                          // SPT (min)
                case 1: key = -dur[j][o]; break;                         // LPT (min of neg)
                case 2: key = -rem[j]; break;                            // MWKR (min of neg)
                case 3: key = max(jobReady[j], bayFree[m]) + dur[j][o]; break; // EFT (min)
                default: key = 0; break;
            }
            if (rule == 4) { cand.push_back(j); continue; }
            if (first || key < bkey) { bkey = key; bj = j; first = false; }
        }
        if (rule == 4) bj = cand[rng() % cand.size()];
        int o = nxt[bj], m = bay[bj][o];
        ll start = max(jobReady[bj], bayFree[m]);
        outSt[bj][o] = start;
        ll end = start + dur[bj][o];
        bayFree[m] = end; jobReady[bj] = end; rem[bj] -= dur[bj][o]; nxt[bj]++;
    }
    ll F = 0;
    for (int j = 0; j < J; j++) for (int o = 0; o < M; o++) F = max(F, outSt[j][o] + dur[j][o]);
    return F;
}

int main() {
    if (scanf("%d %d", &J, &M) != 2) return 0;
    bay.assign(J, vector<int>(M));
    dur.assign(J, vector<ll>(M));
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++) {
            int b; ll d; scanf("%d %lld", &b, &d); bay[j][o] = b; dur[j][o] = d;
        }

    vector<vector<ll>> bestSt, cur;
    ll bestF = LLONG_MAX;

    // deterministic rules
    for (int rule = 0; rule <= 3; rule++) {
        ll F = buildSchedule(rule, cur);
        if (F < bestF) { bestF = F; bestSt = cur; }
    }
    // randomized restarts
    int restarts = 8;
    for (int r = 0; r < restarts; r++) {
        ll F = buildSchedule(4, cur);
        if (F < bestF) { bestF = F; bestSt = cur; }
    }

    for (int j = 0; j < J; j++) {
        for (int o = 0; o < M; o++) {
            printf("%lld", bestSt[j][o]);
            if (o + 1 < M) printf(" ");
        }
        printf("\n");
    }
    return 0;
}
