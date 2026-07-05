// TIER: strong
// Giffler-Thompson active-schedule generation with Most-Work-Remaining (MWKR) priority.
// At each step: among all currently schedulable stages find the one with the earliest
// possible completion c* on some unit m*; then among schedulable stages on m* that could
// start before c*, pick the one whose loop has the most remaining work (tie: earliest
// start, then loop index). This produces compact active schedules that dominate the rigid
// list-scheduling greedy and adapts per-instance (its choices depend on the durations),
// so its per-test makespans genuinely differ.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<vector<int>> mach(N, vector<int>(M)), dur(N, vector<int>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            scanf("%d %d", &mach[i][j], &dur[i][j]);

    vector<int> nxt(N, 0);              // next stage index to schedule for each loop
    vector<long long> jobReady(N, 0), machFree(M, 0);
    vector<long long> remWork(N, 0);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) remWork[i] += dur[i][j];

    vector<vector<long long>> st(N, vector<long long>(M));

    int total = N * M;
    for (int done = 0; done < total; done++) {
        // find earliest completion among schedulable stages
        long long bestC = LLONG_MAX;
        int bestU = -1;
        for (int i = 0; i < N; i++) {
            int j = nxt[i];
            if (j >= M) continue;
            int u = mach[i][j];
            long long s = max(jobReady[i], machFree[u]);
            long long c = s + dur[i][j];
            if (c < bestC) { bestC = c; bestU = u; }
        }
        // conflict set on bestU: schedulable stages that can start before bestC
        int chosen = -1;
        long long chosenStart = 0;
        long long chosenPri = -1, chosenTieStart = LLONG_MAX;
        for (int i = 0; i < N; i++) {
            int j = nxt[i];
            if (j >= M) continue;
            int u = mach[i][j];
            if (u != bestU) continue;
            long long s = max(jobReady[i], machFree[u]);
            if (s >= bestC) continue;          // cannot start before the critical completion
            long long pri = remWork[i];        // MWKR
            if (pri > chosenPri ||
                (pri == chosenPri && (s < chosenTieStart ||
                 (s == chosenTieStart && (chosen == -1 || i < chosen))))) {
                chosen = i; chosenStart = s; chosenPri = pri; chosenTieStart = s;
            }
        }
        // schedule chosen stage
        int j = nxt[chosen];
        int u = mach[chosen][j];
        st[chosen][j] = chosenStart;
        long long e = chosenStart + dur[chosen][j];
        jobReady[chosen] = e;
        machFree[u] = e;
        remWork[chosen] -= dur[chosen][j];
        nxt[chosen]++;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if (j) printf(" ");
            printf("%lld", st[i][j]);
        }
        printf("\n");
    }
    return 0;
}
