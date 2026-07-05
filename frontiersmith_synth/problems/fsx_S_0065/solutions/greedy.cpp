// TIER: greedy
// Single-rule list scheduling (SPT dispatch): repeatedly commit the ready head
// operation with the shortest processing time, placing it at
// max(job_ready, bay_free). Produces a parallel schedule far below serial.
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    vector<vector<int>> bay(J, vector<int>(M));
    vector<vector<ll>> dur(J, vector<ll>(M));
    for (int j = 0; j < J; j++)
        for (int o = 0; o < M; o++) {
            int b; ll d; scanf("%d %lld", &b, &d); bay[j][o] = b; dur[j][o] = d;
        }

    vector<int> nxt(J, 0);
    vector<ll> jobReady(J, 0), bayFree(M + 1, 0);
    vector<vector<ll>> st(J, vector<ll>(M, 0));

    long long totalOps = (long long)J * M;
    for (long long done = 0; done < totalOps; done++) {
        // pick ready head with smallest duration (SPT), tie -> smallest job id
        int bj = -1; ll bd = LLONG_MAX;
        for (int j = 0; j < J; j++) {
            if (nxt[j] >= M) continue;
            ll d = dur[j][nxt[j]];
            if (d < bd) { bd = d; bj = j; }
        }
        int o = nxt[bj], m = bay[bj][o];
        ll start = max(jobReady[bj], bayFree[m]);
        st[bj][o] = start;
        ll end = start + dur[bj][o];
        bayFree[m] = end; jobReady[bj] = end; nxt[bj]++;
    }

    for (int j = 0; j < J; j++) {
        for (int o = 0; o < M; o++) {
            printf("%lld", st[j][o]);
            if (o + 1 < M) printf(" ");
        }
        printf("\n");
    }
    return 0;
}
