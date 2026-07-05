// TIER: greedy
// Fixed-order list scheduling: process stages by stage-index (all loops' stage 1, then
// stage 2, ...). Each stage is placed at the earliest feasible minute given its loop's
// readiness and its unit's next free minute. Exploits parallelism across units, so it
// beats the serial baseline, but the rigid stage-index order leaves avoidable idle gaps.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int N, M;
    if (scanf("%d %d", &N, &M) != 2) return 0;
    vector<vector<int>> mach(N, vector<int>(M)), dur(N, vector<int>(M));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            scanf("%d %d", &mach[i][j], &dur[i][j]);

    vector<long long> jobReady(N, 0), machFree(M, 0);
    vector<vector<long long>> st(N, vector<long long>(M));

    for (int j = 0; j < M; j++) {
        for (int i = 0; i < N; i++) {
            int u = mach[i][j];
            long long s = max(jobReady[i], machFree[u]);
            st[i][j] = s;
            long long e = s + dur[i][j];
            jobReady[i] = e;
            machFree[u] = e;
        }
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
