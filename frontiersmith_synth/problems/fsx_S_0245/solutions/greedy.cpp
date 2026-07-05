// TIER: greedy
// Single non-delay dispatch rule: repeatedly pick, among each inverter's next stage, the one
// with the earliest possible start (tie-break shortest duration). Assign it at the earliest
// feasible time given bay availability and inverter precedence. Always feasible.
#include <bits/stdc++.h>
using namespace std;

int main() {
    int J, M;
    if (scanf("%d %d", &J, &M) != 2) return 0;
    vector<vector<int>> mach(J), dur(J);
    for (int i = 0; i < J; i++) {
        int L; scanf("%d", &L);
        mach[i].resize(L); dur[i].resize(L);
        for (int k = 0; k < L; k++) scanf("%d %d", &mach[i][k], &dur[i][k]);
    }

    vector<vector<long long>> start(J);
    for (int i = 0; i < J; i++) start[i].assign(mach[i].size(), 0);

    vector<int> nextOp(J, 0);
    vector<long long> jobReady(J, 0);
    vector<long long> bayFree(M + 1, 0);

    int totalOps = 0;
    for (int i = 0; i < J; i++) totalOps += (int)mach[i].size();

    for (int done = 0; done < totalOps; done++) {
        int best = -1;
        long long bestS = LLONG_MAX, bestD = LLONG_MAX;
        for (int i = 0; i < J; i++) {
            int k = nextOp[i];
            if (k >= (int)mach[i].size()) continue;
            int m = mach[i][k];
            long long s = max(jobReady[i], bayFree[m]);
            long long d = dur[i][k];
            if (s < bestS || (s == bestS && d < bestD)) {
                bestS = s; bestD = d; best = i;
            }
        }
        int i = best, k = nextOp[i], m = mach[i][k];
        long long s = max(jobReady[i], bayFree[m]);
        start[i][k] = s;
        bayFree[m] = s + dur[i][k];
        jobReady[i] = s + dur[i][k];
        nextOp[i]++;
    }

    for (int i = 0; i < J; i++)
        for (size_t k = 0; k < start[i].size(); k++)
            printf("%lld%c", start[i][k], k + 1 == start[i].size() ? '\n' : ' ');
    return 0;
}
